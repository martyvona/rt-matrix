import { default as Matrix, AbstractMatrix } from 'ml-matrix';
import Random from './random.js';

export default class RTMatrix extends Matrix {

  static asRTMatrix(m, rows, cols) {
    if (m === null || m === undefined) {
      if (cols !== undefined) return new RTMatrix(rows, cols);
      else throw new Error("undefined matrix");
    } else if (m instanceof RTMatrix) {
      if (cols === undefined || (m.rows === rows && m.columns === cols)) return m;
      else throw new Error(`expected ${rows}x${cols} matrix, got ${m.rows}x${m.columns}`);
    } else if (Matrix.isMatrix(m)) {
      if (cols === undefined || (m.rows === rows && m.columns === cols)) return RTMatrix.wrap(m); //TODO
      else throw new Error(`cannot wrap ${m.rows}x${m.columns} Matrix as ${rows}x${cols} RTMatrix`);
    } else if (isAnyArray(m)) {
      if (cols === undefined || m.length === rows * cols) return RTMatrix.wrap(m);
      else throw new Error(`cannot wrap {m.length} array ${rows}x${cols} RTMatrix`);
    } throw new Error(`cannot wrap ${m} as RTMatrix`);
  }

  static getTempArray(i, length) {
    if (!RTMatrix.tempArrays) RTMatrix.tempArrays = [];
    if (!RTMatrix.tempArrays[i] || RTMatrix.tempArrays[i].length < length) {
      RTMatrix.tempArrays[i] = new Float64Array(length);
    }
    return RTMatrix.tempArrays[i];
  }

  constructor(rows, cols) { super(rows, cols); }

  clone() { return RTMatrix.copy(this, new RTMatrix(this.rows, this.columns)); }

  //wraps Matrix.mmul(), for comparison testing
  mmulOrig(b) { return super.mmul(b); }

  //computes c = this * b; if c is undefined it will be allocated
  //same algorithm as mmulOrig() but 
  //* accepts a preallocated c as an optional parameter
  //* explicitly checks matrix dimensions
  //* uses RTMatrix.getTempArray() instead of new
  mmul(b, c) {
    b = RTMatrix.asRTMatrix(b);
    c = RTMatrix.asRTMatrix(c, this.rows, b.columns);
    const ar = this.rows, ac = this.columns;
    const br = b.rows, bc = b.columns;
    const cr = c.rows, cc = c.columns;
    const n = ar, m = ac, p = bc;
    if (br != m || cr != n || cc != p) {
      throw new Error(`cannot multiply ${ar}x${ac} * ${br}x${bc} into ${cr}x${cc}`);
    }
    const tmp = RTMatrix.getTempArray(0, m); //faster to copy out columns of b first, probably due to cache behavior
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < m; k++) tmp[k] = b.get(k, j); //TODO only if b is not a Vector
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let k = 0; k < m; k++) sum += this.get(i, k) * tmp[k];
        c.set(i, j, sum);
      }
    }
    return c;
  }

  //computes c = this * b; if c is undefined it will be allocated
  //this is a wrapper around Matrix.mmulStrassen(), but
  //* accepts a preallocated c as an optional parameter
  //* explicitly checks matrix dimensions - apparently Matrix.mmulStrassen() is only implemented for square matrices
  mmulStrassen(b, c) {
    b = RTMatrix.asRTMatrix(b);
    c = RTMatrix.asRTMatrix(c, this.rows, b.columns);
    const ar = this.rows, ac = this.columns;
    const br = b.rows, bc = b.columns;
    const cr = c.rows, cc = c.columns;
    if (ar != ac || br != bc || ar != br) {
      throw new Error(`cannot multiply ${ar}x${ac} * ${br}x${bc} into ${cr}x${cc}: ` +
                      `mmulStrassen() is only implemented for square matrices`);
    }
    RTMatrix.copy(super.mmulStrassen(b), c);
    return c;
  }

  //computes c = this * b; if c is undefined it will be allocated
  mmulNaive(b, c) {
    b = RTMatrix.asRTMatrix(b);
    c = RTMatrix.asRTMatrix(c, this.rows, b.columns);
    const ar = this.rows, ac = this.columns;
    const br = b.rows, bc = b.columns;
    const cr = c.rows, cc = c.columns;
    const n = ar, m = ac, p = bc;
    if (br != m || cr != n || cc != p) {
      throw new Error(`cannot multiply ${ar}x${ac} * ${br}x${bc} into ${cr}x${cc}`);
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < m; k++) sum += this.get(i, k) * b.get(k, j);
        c.set(i, j, sum);
      }
    }
    return c;
  }

  //computes c = this * b; if c is undefined it will be allocated
  //this is an implementation of the recursive block algorithm described here:
  //https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Non-square_matrices
  //however in tests up to 500x500 this doesn't seem to perform as well as mmul()
  //and also seems to have worse numeric precision
  //also, baseSize may need to be tuned
  mmulBlock(b, c) {
    b = RTMatrix.asRTMatrix(b);
    c = RTMatrix.asRTMatrix(c, this.rows, b.columns);
    const baseSize = 100;
    const a = this;
    function mmulSub(arl, aru, acl, acu, brl, bru, bcl, bcu, crl, cru, ccl, ccu, accum) {
      const ar = aru - arl + 1, ac = acu - acl + 1;
      const br = bru - brl + 1, bc = bcu - bcl + 1;
      const cr = cru - crl + 1, cc = ccu - ccl + 1;
      const n = ar, m = ac, p = bc;
      if (br != m || cr != n || cc != p) {
        throw new Error(`cannot multiply submatrices ${ar}x${ac} * ${br}x${bc} into ${cr}x${cc}`);
      }
      const tmp = RTMatrix.getTempArray(0, m);
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < m; k++) tmp[k] = b.get(brl + k, bcl + j); //TODO only if b is not a Vector
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let k = 0; k < m; k++) sum += a.get(arl + i, acl + k) * tmp[k];
          if (accum) sum += c.get(crl + i, ccl + j);
          c.set(crl + i, ccl + j, sum);
        }
      }
      return c;
    }
    function mmulRecursive(arl, aru, acl, acu, brl, bru, bcl, bcu, crl, cru, ccl, ccu, accum) {
      const ar = aru - arl + 1, ac = acu - acl + 1;
      const br = bru - brl + 1, bc = bcu - bcl + 1;
      const cr = cru - crl + 1, cc = ccu - ccl + 1;
      const n = ar, m = ac, p = bc;
      if (br != m || cr != n || cc != p) {
        throw new Error(`cannot recursive multiply ${ar}x${ac} * ${br}x${bc} into ${cr}x${cc}`);
      }
      const maxDim = Math.max(n, m, p);
      if (maxDim < baseSize) return mmulSub(arl, aru, acl, acu, brl, bru, bcl, bcu, crl, cru, ccl, ccu, accum);
      const split = Math.floor(0.5 * maxDim);
      if (maxDim === n) { //split A horizontally
        //C = [ Al * B ]
        //    [ Au * B ]
        mmulRecursive(arl,     arl + split, acl, acu, brl, bru, bcl, bcu, crl,      crl + split, ccl, ccu, accum);
        mmulRecursive(arl + split + 1, aru, acl, acu, brl, bru, bcl, bcu, crl + split + 1, cru, ccl, ccu, accum);
      } else if (maxDim === p) { //split B vertically
        //C = [A * Bl, A * Bu]
        mmulRecursive(arl, aru, acl, acu, brl, bru, bcl,     bcl + split, crl, cru, ccl,     ccl + split, accum);
        mmulRecursive(arl, aru, acl, acu, brl, bru, bcl + split + 1, bcu, crl, cru, ccl + split + 1, ccu, accum);
      } else { //split A vertically and B horizontally
        //C = Al * Bl + Au * Bu
        mmulRecursive(arl, aru, acl,     acl + split, brl,     brl + split, bcl, bcu, crl, cru, ccl, ccu, accum);
        mmulRecursive(arl, aru, acl + split + 1, acu, brl + split + 1, bru, bcl, bcu, crl, cru, ccl, ccu, true);
      }
      return c;
    }
    return mmulRecursive(0, a.rows - 1, 0, a.columns - 1,
                         0, b.rows - 1, 0, b.columns - 1,
                         0, c.rows - 1, 0, c.columns - 1, false);
  }
}

export function testMMUL() {

  const minVal = -1e6;
  const maxVal = 1e6;
  const maxDim = 200;
  const numRandomPairs = 20;
  const numShuffles = 10;

  //https://docs.python.org/3/library/math.html#math.isclose
  const relTol = 1e-9, absTol = 0;
  //const relTol = 1e-7, absTol = 0; //needed for mmulBlock()
  const eps = (a, b) => Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
  const isClose = (a, b) => (Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol));

  const rand = Random("test");

  const randMat = (rows, cols) => {
    const m = new RTMatrix(rows, cols);
    for (let i = 0; i < rows; i++) { for (let j = 0; j < cols; j++) m.set(i, j, rand.nextIn(minVal, maxVal)); }
    return m;
  };

  const dims = [[1, 1, 1], [maxDim / 10, 1, maxDim / 10], [maxDim / 10, maxDim / 10, maxDim / 10],
                [maxDim, 1, maxDim], [maxDim, 2, maxDim], [maxDim, maxDim, maxDim]];
  for (let i = 1; i < numRandomPairs; i++) {
    dims.push([rand.nextInt(maxDim), rand.nextInt(maxDim), rand.nextInt(maxDim)]);
  }

  let started, paused, msg;
  const start = (m) => { console.log(`${m}...`); msg = m; started = performance.now(); }
  const stop = () => { const ms = performance.now() - started; console.log(`finished ${msg}: ${ms | 0}ms`); return ms; }
  const pause = () => { paused = performance.now(); }
  const unpause = () => { started += performance.now() - paused; }

  start(`generating ${dims.length * 2} random matrices up to ${maxDim}x${maxDim}`);
  const mats = [];
  for (const d of dims) {
    const n = d[0], m = d[1], p = d[2];
    const a = randMat(n, m);
    const b = randMat(m, p);
    const c = new RTMatrix(n, p);
    const g = null;
    mats.push([a, b, c, g]);
  }
  stop();

  const checkMat = (m, t) => {
    if (t === null) return;
    //console.log(`checking ${m.rows}x${m.columns} matrix`);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.columns; j++) {
        const actual = m.get(i, j), truth = t.get(i, j), diff = actual - truth;
        if (!isClose(actual, truth)) {
          throw new Error(`error at row ${i} col ${j} in ${m.rows}x${m.columns} matrix: ` +
                          `${actual} - ${truth} = ${diff} > ${eps(actual, truth)}`);
        }
      }
    }
  };
  
  const runTest = (mmul, what) => {

    const shuffle = []; for (let j = 0; j < mats.length; j++) shuffle[j] = j;

    start(`performing ${mats.length} ${what} matrix multiplications in ${numShuffles} random orders`);

    for (let i = 0; i < numShuffles; i++) {
      for (const j of shuffle) {
        const a = mats[j][0], b = mats[j][1], c = mats[j][2];
        //console.log(`${what}(${a.rows}x${a.columns}, ${b.rows}x${b.columns}, ${c.rows}x${c.columns})`);
        mmul(a, b, c);
      }

      pause();
      for (let i = 0; i < mats.length; i++) checkMat(mats[i][2], mats[i][3]);
      //console.log(`shuffle ${i}...`);
      for (let j = 0; j < shuffle.length; j++) {
        const k = rand.nextInt(shuffle.length), l = rand.nextInt(shuffle.length);
        const tmp = shuffle[k];
        shuffle[k] = shuffle[l];
        shuffle[l] = tmp;
      }
      unpause();
    }

    return stop();
  }

  RTMatrix.getTempArray(0, maxDim);

  runTest((a, b, c) => a.mmulNaive(b, c), "mmulNaive");
  console.log('saving ground truth');
  for (let i = 0; i < mats.length; i++) mats[i][3] = new RTMatrix(mats[i][2]);

  //runTest((a, b, c) => a.mmulBlock(b, c), "mmulBlock");
  runTest((a, b, c) => RTMatrix.copy(a.mmulOrig(b), c), "mmulOrig");
  runTest((a, b, c) => a.mmul(b, c), "mmul");
  //runTest((a, b, c) => a.mmulStrassen(b, c), "mmulStrassen");

  start(`generating ${numRandomPairs * 2} random matrices exactly ${maxDim}x${maxDim}`);
  mats.length = 0;
  for (let i = 0; i < numRandomPairs; i++) {
    const n = maxDim, m = maxDim, p = maxDim;
    const a = randMat(n, m);
    const b = randMat(m, p);
    const c = new RTMatrix(n, p);
    const g = null;
    mats.push([a, b, c, g]);
  }
  stop();

  const nt = numRandomPairs * numShuffles;
  let ms = runTest((a, b, c) => a.mmulNaive(b, c), "mmulNaive");
  console.log(`avg ${ms/nt}ms per mmulNaive`);
  console.log('saving ground truth');
  for (let i = 0; i < mats.length; i++) mats[i][3] = new RTMatrix(mats[i][2]);
  //ms = runTest((a, b, c) => a.mmulBlock(b, c), "mmulBlock");
  //console.log(`avg ${ms/nt}ms per mmulBlock`);
  ms = runTest((a, b, c) => RTMatrix.copy(a.mmulOrig(b), c), "mmulOrig");
  console.log(`avg ${ms/nt}ms per mmulOrig`);
  ms = runTest((a, b, c) => a.mmul(b, c), "mmul");
  console.log(`avg ${ms/nt}ms per mmul`);
  ms = runTest((a, b, c) => a.mmulStrassen(b, c), "mmulStrassen");
  console.log(`avg ${ms/nt}ms per mmulStrassen`);
}
