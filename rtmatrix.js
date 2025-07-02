// "Real Time" Matrix Library
//
// Marsette Vona, 2024
//
// * supports variable size dense 2D rectangular matrices stored in a Float64Array
// * supports storage pre-allocation to avoid garbage creation for real-time interactive applications
// * column major storage for compatibility with LAPACK and Eigen
// * TODO links to a wasm build of the LAPACK SVD which is both fast and supports storage pre-allocation
//
// Graphics-oriented JavaScript matrix libraries (e.g. glMatrix, glm-js, THREE.js) allow storage pre-allocation, but
// typically only support matrices up to 4x4 and do not implement SVD.
//
// Science-oriented JavaScript matrix libraries (e.g. ml-matrix, Math.js, matrix-js), allow matrices larger than 4x4,
// but do not typically support storage pre-allocation.  These libraries typically do include SVD, but that may not be
// as advanced as in LAPACK or Eigen, e.g. only implementing a Jacobi algorithm.  These libraries also often use
// row-major storage, which would not be directly compatible with LAPACK and Eigen, generally requiring a data copy for
// interoperation with (wasm builds of) those.
//
// Though internal storage is generally column-major, support is also included for specifying literals in row-major
// order as an array of arrays, e.g.
// [[1, 2],
//  [3, 4]]
//
// Methods do not allocate memory (except for reusable temp storage) except as noted.

// a 2D matrix with direct contiguous column major storage
export default class Matrix {

  // check if obj is a Matrix or any of its derived classes
  static isMatrix(obj) { return obj instanceof Matrix; }

  // check if obj is a Matrix with one column or one row
  static isVector(obj) { return Matrix.isMatrix(obj) && (obj.cols === 1 || obj.rows === 1); }

  // make printable dimensions
  static dims(rows, cols) {
    return `[${Number.isInteger(rows) ? String(rows) : "?"}x${Number.isInteger(cols) ? String(cols) : "?"}]`;
  }

  // make sure rows and cols are positive integers
  static checkDims(rows, cols) {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
      throw new RangeError(`invalid dimensions ${Matrix.dims(rows, cols)}`);
    }
  }

  // get or allocate the ith shared temp Float64Array
  // if it has already been allocated but doesn't have at least the required length it will be reallocated
  static getTempArray(i, length) {
    if (!Number.isInteger(i) || i < 0) throw new Error(`invalid index: ${i}`);
    if (!Number.isInteger(length) || length <= 0) throw new Error(`invalid length: ${length}`);
    if (!Matrix.tempArrays) Matrix.tempArrays = [];
    if (!Matrix.tempArrays[i] || Matrix.tempArrays[i].length < length) Matrix.tempArrays[i] = new Float64Array(length);
    return Matrix.tempArrays[i];
  }

  // similar to getTempArray() but returns a shared temp Matrix wrapping data
  static getTempMatrix(i, rows, cols = 1, data = this.getTempArray(i, rows * cols)) {
    Matrix.checkDims(rows, cols);
    if (!Number.isInteger(i) || i < 0) throw new Error(`invalid index: ${i}`);
    if (!Matrix.tempMatrices) Matrix.tempMatrices = [];
    if (!Matrix.tempMatrices[i]) Matrix.tempMatrices[i] = new Matrix(rows, cols, data);
    else return Matrix.tempMatrices[i].init(rows, cols, data);
  }

  // similar to getTempArray() but returns a shared temp RowMatrix wrapping data
  static getTempRowMatrix(i, rows, cols, data) {
    Matrix.checkDims(rows, cols);
    if (!Number.isInteger(i) || i < 0) throw new Error(`invalid index: ${i}`);
    if (!Matrix.tempRowMatrices) Matrix.tempRowMatrices = [];
    if (!Matrix.tempRowMatrices[i]) Matrix.tempRowMatrices[i] = new RowMatrix(rows, cols, data);
    else return Matrix.tempRowMatrices[i].init(rows, cols, data);
  }

  // free the data pointers of all temp matrices
  // does not actually free the matrices themselves
  static clearTempMatrices() {
    if (Matrix.tempMatrices) for (const m of Matrix.tempMatrices) m.data = null;
    if (Matrix.tempRowMatrices) for (const m of Matrix.tempRowMatrices) m.data = null;
  }

  // free all temp arrays and matrices
  static clearAllTemps() {
    Matrix.tempArrays = null;
    Matrix.tempMatrices = null;
    Matrix.tempRowMatrices = null;
  }

  // asTempMatrix(i, null|undefined) throws Error
  // asTempMatrix(i, matrix) returns matrix
  // asTempMatrix(i, matrix, rows[, cols]) returns matrix iff it has the indicated size
  // asTempMatrix(i, array) returns a temp Matrix wrapping array as a column vector
  // asTempMatrix(i, array, rows) returns a temp Matrix wrapping array as a column vector iff array.length >= rows
  // asTempMatrix(i, array, rows, cols) returns a temp Matrix wrapping column-major data iff array.length >= rows * cols
  // asTempMatrix(i, arrayOfArrays) returns a temp Matrix wrapping row-major data in arrayOfArrays
  // asTempMatrix(i, arrayOfArrays, rows[, cols]) returns temp Matrix wrapping row-major data iff it has given size
  static asTempMatrix(i, obj, rows, cols) {
    if (obj === null || obj === undefined) {
      throw new Error("null or undefined argument");
    } else if (obj instanceof Matrix) {
      if ((rows === undefined || rows === obj.rows) && (cols === undefined || cols === obj.cols)) return obj;
      else throw new RangeError(`expected ${Matrix.dims(rows, cols)}, got ${obj.dims}`);
    } else if (isAnyArray(obj)) {
      if (rows === undefined) rows = obj.length;
      if (rows > 0 && isAnyArray(obj[0])) {
        if (cols === undefined) cols = obj[0].length;
        if (obj.length === rows && obj[0].length === cols) return Matrix.getTempRowMatrix(i, rows, cols, obj);
        else throw new RangeError
        (`cannot wrap ${Matrix.dims(obj.length, obj[0].length)} array of arrays as ${Matrix.dims(rows, cols)}`);
      } else {
        if (cols === undefined) cols = 1;
        if (rows * cols <= obj.length) return Matrix.getTempMatrix(i, rows, cols, obj);
        else throw new RangeError(`cannot wrap ${obj.length} array as ${Matrix.dims(rows, cols)}`);
      }
    } throw new Error(`cannot convert ${obj} to Matrix`);
  }

  // returns a new matrix of the form
  // [ A ]
  // [ B ]
  // [ . ]
  // [ . ]
  // [ . ]
  // which reads and writes through to the arguments, all of which must have the same widths
  // NOTE: allocates memory
  static stacked(...matrices) { return new StackedMatrix(matrices); }

  // returns a new matrix of the form [ A B . . . ]
  // which reads and writes through to the arguments, all of which must have the same heights
  // NOTE: allocates memory
  static glued(...matrices) { return new GluedMatrix(matrices); }

  // new Matrix(rows, cols) allocates a new Float64Array for storage, initialized to 0
  // new Matrix(rows, cols, array) uses the given array of column major data iff it's big enough
  constructor(rows, cols, data) { this.init(rows, cols, data); }

  // (re-)initialize this matrix, if possible
  // NOTE: allocates memory if data is undefined
  init(rows, cols, data) {
    Matrix.checkDims(rows, cols);
    this.rows = rows;
    this.cols = cols;
    if (data == undefined) data = new Float64Array(rows * cols);
    else if (!isAnyArray(data) || data.length < rows * cols) {
      throw new RangeError(`cannot use ${data} as storage for ${this.dims} matrix`);
    }
    this.data = data;
    return this;
  }

  // returns the shape and type of this matrix
  toString() { return `${this.dims} ${this.constructor.name}`; }

  // get printable dimensions of this matrix
  get dims() { return Matrix.dims(this.rows, this.cols); }

  // return the number of elements in this matrix
  get length() { return this.rows * this.cols; }

  // check if the underlying storage of this matrix is in column major order
  isColumnMajor() { return true; }

  // check if the underlying storage of this matrix is a Float64Array
  isFloat64() { return this.data instanceof Float64Array; }

  // check if getData() can be called
  isDirect() { return true; }

  // check if array returned from getData() is contiguous
  isContiguous() { return true; }

  // get the underlying storage array of this matrix, if possible
  getData() { return this.data; }

  // reshape this matrix in place, if possible
  // the new length (rows * cols) must be less than or equal to the current size of the underlying storage array
  // the contents of the underlying storage are not modified, so the contents of the resulting matrix
  // will be the existing contents treated as a vector in column major order
  // plus any unknown data beyond that if the new length (rows * cols) is greater than the current length
  // and then reinterpreted as a matrix with the new dimensions in column major order
  reshape(rows, cols = this.cols) {
    Matrix.checkDims(rows, cols);
    if (rows * cols > this.data.length) {
      throw new RangeError(`cannot reshape ${this.dims} as ${Matrix.dims(rows, cols)}: ` +
                           `${rows * cols} > ${this.data.length}`);
    }
    this.rows = rows;
    this.cols = cols;
    return this;
  }

  // return a new matrix that reads and writes through to this matrix treated as a vector in column major order
  // and then reinterpreted as a matrix with the specified dimensions
  // if cols is omitted it defaults to the number of columns of this matrix
  // the new length (rows * cols) must be less than or equal to the current length
  // NOTE: allocates memory if ret is undefined
  reshaped(rows, cols = this.cols, ret) {
    return ret === undefined ? new ReshapedMatrix(rows, cols, this) : ret.init(rows, cols, this);
  }

  // reallocate the underlying storage array to the specified size
  // has no effect if the array is already that size
  // otherwise, the storage array is replaced with a new Float64Array
  // the new size must be at least as large as the curent length (rows * cols) of this matrix
  // returns this matrix
  reallocate(size) {
    if (!Number.isInteger(size) || size < 0) throw new Error(`invalid size: ${size}`);
    if (this.data.length === size) return;
    const len = this.length;
    if (size < len) throw new RangeError(`cannot reallocate matrix with ${len} elements to ${size}`);
    const newData = new Float64Array(size);
    for (let i = 0; i < len; i++) newData[i] = this.data[i];
    this.data = newData;
    return this;
  }

  // reallocate() iff size is greater than the current size of the underlying storage array; returns this matrix
  ensureStorage(size) { return size > this.data.length ? reallocate(size) : this; }

  // double the size of the underlying storage array; returns this matrix
  grow() { return reallocate(2 * this.data.length); }

  // reallocate() to exactly the curent length (rows * cols) of this matrix; returns this matrix
  shrink() { return reallocate(this.length); }

  // get or set an an element of this matrix
  // all other operations are implemented in terms of get() and set(), except where noted
  get(row, col)      { return this.data[col * rows + row]; }
  set(row, col, val) {        this.data[col * rows + row] = val; return this; }

  // get or set an element of this matrix treated as a vector in column major order
  at(index)       { return this.data[index]; }
  put(index, val) {        this.data[index] = val; return this; }

  // get a reference to the contiguous Float64 column major data array of this matrix, if possible, otherwise
  // * if ret is a Float64Array with length at least as large as the number of entries in this matrix, copy
  //   this matrix to it in column major form and return it
  // * if ret is an integer then return the data in column major order in Matrix.getTempArray(ret, this.length)
  getFloat64ColumnMajorData(ret) { 
    if (this.isColumnMajor() && this.isFloat64() && this.isDirect() && this.isContiguous()) return this.getData();
    else if (ret instanceof Float64Array && ret.length >= this.length) return this.copyTo(ret);
    else if (Number.isInteger(ret)) return this.copyTo(Matrix.getTempArray(ret, this.length));
    else throw new Error(`cannot copy ${this.rows}x${this.cols} to ${ret}`);
  }

  // get/set the first four elements of this matrix treated as a vector in column major order
  get x()    { return this.at(0); }
  set x(val) {        this.put(0, val); }
  get y()    { return this.at(1); }
  set y(val) {        this.put(1, val); }
  get z()    { return this.at(2); }
  set z(val) {        this.put(2, val); }
  get w()    { return this.at(3); }
  set w(val) {        this.put(3, val); }

  // returns a new matrix that reads and writes through to a sub-matrix of this matrix
  // NOTE: allocates memory if ret is undefined
  slice(startRow = 0, numRows = this.rows - startRow, startCol = 0, numCols = this.cols - startCol, ret) {
    if (ret === undefined) return new SlicedMatrix(startRow, numRows, startCol, numCols, this);
    else return ret.reset(startRow, numRows, startCol, numCols, this);
  }

  // returns a new matrix that reads and writes through to a selected set of rows and columns of this matrix
  // NOTE: allocates memory if colIndices or ret is undefined
  select(rowIndices, colIndices = (new Array(this.cols)).fill(0).map((_,i) => i), ret) {
    if (ret === undefined) return new SelectedMatrix(rowIndices, colIndices, this);
    else return ret.reset(rowIndices, colIndices, this);
  }

  // sugar for select()
  // returns a new matrix that reads and writes through to a selected set of rows and a column range of this matrix
  // NOTE: allocates memory
  selectRows(rowIndices, startCol = 0, numCols = this.cols - startCol, ret) {
    return select(rowIndices, (new Array(numCols)).fill(0).map((_,i) => startCol + i), this);
  }

  // sugar for select()
  // returns a new matrix that reads and writes through to a selected set of columns and a row range of this matrix
  // NOTE: allocates memory
  selectCols(colIndices, startRow = 0, numRows = this.rows - startRow, ret) {
    return select((new Array(numRows)).fill(0).map((_,i) => startrow + i), colIndices, this);
  }

  // fill a submatrix with a constant scalar value; returns this
  fill(val, dstRow = 0, numRows = this.rows - dstRow, dstCol = 0, numCols = this.cols - dstCol) {
    const m = numRows, n = numCols;
    if (dstRow < 0 || dstCol < 0 || m < 0 || n < 0 || this.rows < dstRow + m || this.cols < dstCol + n) {
      throw new RangeError(`cannot fill ${Matrix.dims(m, n)} of ${this.dims} at (${dstRow}, ${dstCol}) with ${val}`);
    }
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < m; r++) this.set(dstRow + r, dstCol + c, val);
    }
    return this;
  }

  // sugar for fill(0, dstRow, numRows, dstCol, numCols)
  zero(dstRow, numRows, dstCol, numCols) { return this.fill(0, dstRow, numRows, dstCol, numCols); }

  // sugar for zero(dstRow, numRows, dstCol, numCols)
  clear(dstRow, numRows, dstCol, numCols) { return this.zero(dstRow, numRows, dstCol, numCols); }

  // fill a square submatrix of this matrix with the identity matrix
  // dstRow and dstCol default to 0
  // size defaults to the minimum dimension of this matrix minus the corresponding offset
  eye(dstRow = 0, dstCol = 0, size = Math.min(this.rows - dstRow, this.cols - dstCol)) {
    const n = size;
    if (dstRow < 0 || dstCol < 0 || n < 0 || this.rows < dstRow + n || this.cols < dstCol + n) {
      throw new RangeError(`cannot fill ${Matrix.dims(n, n)} of ${this.dims} at (${dstRow}, ${dstCol}) with identity`);
    }
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < n; r++) this.set(dstRow + r, dstCol + c, r === c ? 1 : 0);
    }
    return this;
  }

  // copies a submatrix of an array or matrix to this matrix
  //
  // if src is an array of arrays it's treated as a row matrix
  // otherwise, if src is an array it's treated as column major data
  // in either of those cases, srcRow and srcCol must be 0
  // otherwise, src must be a matrix
  //
  // numRows and numCols default to the minimum size remaining in either matrix in the corresponding dimension
  //
  // returns this
  copyFrom(src, dstRow = 0, numRows, srcRow = 0, dstCol = 0, numCols, srcCol = 0) {
    const srcIsArray = isAnyArray(src);
    if (!(isMatrix(src) || srcIsArray) || src === this) throw new Error(`cannot copy ${src} to ${this.dims}`);
    let m = numRows, n = numCols;
    if (srcIsArray) {
      if (srcRow !== 0 || srcCol !== 0) throw new Error(`cannot copy from array at $(${srcRow}, ${srcCol})`);
      if (m === undefined) m = this.rows - dstRow;
      if (n === undefined) n = this.cols - dstcol;
      src = asTempMatrix(0, src, m, n);
    } else {
      if (m === undefined) m = Math.min(this.rows - dstRow, src.rows - srcRow);
      if (n === undefined) n = Math.min(this.cols - dstCol, src.cols - srcCol);
    }
    try {
      if (dstRow < 0 || dstCol < 0 || m < 0 || n < 0 || this.rows < dstRow + m || this.cols < dstCol + n) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} to ${this.dims} at (${dstRow}, ${dstCol})`);
      }
      if (srcRow < 0 || srcCol < 0 || src.rows < srcRow + m || src.cols < srcCol + n) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} from ${src.dims} at (${srcRow}, ${srcCol})`);
      }
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) this.set(dstRow + r, dstCol + c, src.get(srcRow + r, srcCol + c));
      }
      return this;
    } finally { Matrix.clearTempMatrices(); }
  }

  // copies a submatrix of this matrix to dst
  //
  // if dst is an array both it and this matrix are treated as a column vectors (column major order)
  // otherwise dst must be a matrix
  //
  // numRows and numCols default to the minimum size remaining in either matrix in the corresponding dimension
  //
  // returns dst
  copyTo(dst, dstRow = 0, numRows, srcRow = 0, dstCol = 0, numCols, srcCol = 0) {
    const dstIsArray = isAnyArray(dst);
    if (!(Matrix.isMatrix(dst) || dstIsArray) || dst === this) throw new Error(`cannot copy ${src} to ${dst}`)
    if (dstIsArray) {
      const m = numRows === undefined ? this.rows - srcRow : numRows;
      const n = numCols === undefined ? this.cols - srcCol : numCols;
      if (dstRow < 0 || dstCol != 0 || m < 0 || n < 0 || dst.length < dstRow + (m * n)) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} to ${dst.length} vector at ${dstRow}`);
      }
      if (srcRow < 0 || srcCol < 0 || this.rows < srcRow + m || this.cols < srcCol + n) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} from ${this.dims} at (${srcRow}, ${srcCol})`);
      }
      let i = dstRow;
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) dst[i++] = this.get(srcRow + r, srcCol + c);
      }
    } else {
      const m = numRows === undefined ? Math.min(this.rows - srcRow, dst.rows - dstRow) : numRows;
      const n = numCols === undefined ? Math.min(this.cols - srcCol, dst.cols - dstCol) : numCols;
      if (dstRow < 0 || dstCol < 0 || m < 0 || n < 0 || dst.rows < dstRow + m || dst.cols < dstCol + n) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} to ${dst.dims} at (${dstRow}, ${dstCol})`);
      }
      if (srcRow < 0 || srcCol < 0 || this.rows < srcRow + m || this.cols < srcCol + n) {
        throw new RangeError(`cannot copy ${Matrix.dims(m, n)} from ${this.dims} at (${srcRow}, ${srcCol})`);
      }
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) dst.set(dstRow + r, dstCol + c, this.get(srcRow + r, srcCol + c));
      }
    }
    return dst;
  }

  // computes the dot product of this matrix with other
  //
  // if other is an array of arrays it's treated as a row matrix
  // otherwise if other is an array it's treated as a column vector
  // otherwise other must be a matrix
  //
  // this and other are then treated as column vectors (column major order), and must have same lenghths (rows * cols)
  dot(other) {
    try {
      other = Matrix.asTempMatrix(0, other);
      const n = this.length;
      if (other.length !== n) throw new RangeError(`cannot compute dot product of ${this.dims} and ${other.dims}`);
      let sum = 0;
      for (let i = 0; i < n; i++) sum += this.at(i) * other.at(i);
      return sum;
    } finally { Matrix.clearTempMatrices(); }
  }

  // returns the Euclidean (i.e. L^2) norm of this matrix
  norm() { return Math.sqrt(this.dot(this)); }

  // computes dst = this x other
  //
  // if other is an array of arrays it's treated as a row matrix
  // otherwise if other is an array it's treated as a column vector
  // otherwise other must be a matrix
  //
  // dst must be a matrix
  //
  // this matrix, other, and dst must all have length (rows * cols) equal to 3
  //
  // returns dst
  cross(other, dst = this) {
    try {
      if (this.length !== 3) throw new RangeError(`cannot compute cross product of ${this.dims}`);

      other = Matrix.asTempMatrix(0, other);
      if (other.length !== 3) throw new RangeError(`cannot compute cross product with ${other.dims}`);
      
      if (!isMatrix(dst)) throw new Error(`cannot compute cross product into ${dst}`);
      else if (dst.length !== 3) throw new RangeError(`cannot compute cross product into ${dst.dims}`);
      
      const tx = this.x, ty = this.y, tz = this.z;
      const ox = other.x, oy = other.y, oz = other.z;
      
      //tx ty tz
      //ox oy oz
      dst.x = ty * oz - tz * oy;
      dst.y = tz * ox - tx * oz;
      dst.z = tx * oy - ty * ox;
      
      return dst;
    } finally { Matrix.clearTempMatrices(); }
  }

  // computes dst = thisScale * this + otherScale * other
  //
  // if other is an array of arrays it's treated as a row matrix
  // otherwise if other is an array it's treated as a column vector
  // otheriwse if other is a scalar it's treated as a matrix of the same shape as this matrix
  //
  // dst must be a matrix
  //
  // other and dst must have the same shape as this matrix
  //
  // returns dst
  mulAdd(thisScale, other, otherScale, dst = this) {
    const m = this.rows, n = this.cols;
    if (!isMatrix(dst)) throw new Error(`cannot mulAdd ${this.dims} into ${dst}`);
    else if (dst.rows !== this.rows || dst.cols !== this.cols) {
      throw new RangeError(`cannot mulAdd ${this.dims} into ${dst.dims}`);
    }
    if (typeof other === "number") {
      other *= otherScale;
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) dst.set(thisScale * this.get(r, c) + other);
      }
    } else {
      try {
        other = Matrix.asTempMatrix(0, other, m, n);
        for (let c = 0; c < n; c++) {
          for (let r = 0; r < m; r++) dst.set(thisScale * this.get(r, c) + otherScale * other.get(r, c));
        }
      } finally { Matrix.clearTempMatrices(); }
    }
    return dst;
  }

  // similar to mulAdd() but computes dst = this + other
  add(other, dst = this) { return mulAdd(1, other, 1, dst); }

  // similar to mulAdd() but computes dst = this - other
  sub(other, dst = this) { return mulAdd(1, other, -1, dst); }

  // similar to mulAdd() but computes dst = other - this
  subFrom(other, dst = this) { return mulAdd(-1, other, 1, dst); }

  // componentwise multiply this matrix by other and store the result in dst
  //
  // if other is a scalar it's treated as a matrix of the same shape as this matrix
  // otherwise if other is an array of arrays it's treated as a row matrix
  // otherwise if other is an array it's treated as a column vector
  // otherwise other must be a matrix
  //
  // dst must be a matrix
  //
  // other and dst must have the same shape as this matrix
  //
  // returns dst
  scale(other, dst = this) {
    const m = this.rows, n = this.cols;
    if (!isMatrix(dst)) throw new Error(`cannot scale ${this.dims} into ${dst}`);
    else if (dst.rows !== this.rows || dst.cols !== this.cols) {
      throw new RangeError(`cannot scale ${this.dims} into ${dst.dims}`);
    }
    if (typeof other === "number") {
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) dst.set(this.get(r, c) * other);
      }
    } else {
      try {
        other = Matrix.asTempMatrix(0, other, m, n);
        for (let c = 0; c < n; c++) {
          for (let r = 0; r < m; r++) dst.set(this.get(r, c) * other.get(r, c));
        }
      } finally { Matrix.clearTempMatrices(); }
    }
    return dst;
  }

  // compute dst = this^T
  //
  // if dst is this matrix it must either be square or a column vector or row vector
  // otherwise dst must be an n x m matrix where this matrix is m x n
  //
  // returns dst
  transpose(dst = this) {
    const m = this.rows, n = this.cols;
    if (dst === this) {
      if (m > 1 && n > 1) {
        //it is possible to do a general in-place rectangular transpose, but it's relatively complex and expensive
        if (m !== n) throw new RangeError(`cannot transpose ${this.dims} in place`);
        for (let c = 0; c < n; c++) {
          for (let r = c + 1; r < m; r++) {
            const tmp = this.get(r, c);
            this.set(r, c, this.get(c, r));
            this.set(c, r, tmp);
          }
        }
      }
      this.rows = n;
      this.cols = m;
    } else if (isMatrix(dst) && dst.rows === n && dst.cols === m) { 
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) dst.set(r, c, this.get(c, r));
      }
    } else throw new Error(`cannot transpose ${this.dims} into ${dst}`);
    return dst;
  }

  // compute dst = this * other
  //
  // if other is an array of arrays it's treated as a row matrix
  // otherwise if other is an array it's treated as a column vector
  // otherwise other must be a matrix
  //
  // this.cols must equal other.rows
  //
  // dst must be a matrix where dst.rows equals this.rows and dst.cols equals other.cols
  //
  // if dst is this matrix
  // * an internal temporary copy is made
  // * other must be square and have the same size as the number of columns as this matrix
  //
  // returns dst
  mul(other, dst = this) {
    try {
      const m = this.rows, n = this.cols;
      other = Matrix.asTempMatrix(0, other, n);
      const p = other.cols;
      
      if (!isMatrix(dst)) throw new Error(`cannot multiply into ${dst}`);
      else if (dst.rows !== m || dst.cols !== p) {
        throw new RangeError(`cannot multiply ${this.dims} * ${other.dims} into ${dst.dims}`);
      }
      
      if (dst === this) {
        const rowMajorCopy = Matrix.getTempArray(1, m * n); //can't multiply in-place; row maj to be CPU cache friendly
        for (let c = 0; c < n; c++) {
          for (let r = 0; r < m; r++) rowMajorCopy[r * n + c] = this.get(r, c);
        }
        for (let c = 0; c < p; c++) {
          for (let r = 0; r < m; r++) {
            let sum = 0;
            for (let i = 0; i < n; i++) sum += rowMajorCopy[r * n + i] * other.get(i, c);
            dst.set(r, c, sum);
          }
        }
      } else if (m === 1) { //[1 x n] * [n x p]
        for (let c = 0; c < p; c++) {
          let sum = 0;
          for (let i = 0; i < n; i++) sum += this.get(0, c) * other.get(i, c);
          dst.set(0, c, sum);
        }
      } else if (n > 1) { //[m x n] * [n x p] including [m x n] * [n x 1]
        if (m <= 4 && n <= 4) {
          for (let r = 0; r < m; r++) {
            for (let c = 0; c < p; c++) {
              let sum = 0;
              for (let i = 0; i < n; i++) sum += this.get(r, i) * other.get(i, c);
              dst.set(r, c, sum);
            }
          }
        } else {
          const tmpRow = Matrix.getTempArray(1, n); //for CPU cache friendlyness
          for (let r = 0; r < m; r++) {
            for (let i = 0; i < n; i++) tmpRow[i] = this.get(r, i);
            for (let c = 0; c < p; c++) {
              let sum = 0;
              for (let i = 0; i < n; i++) sum += tmpRow[i] * other.get(i, c);
              dst.set(r, c, sum);
            }
        }
      } else { //[m x 1] * [1 x p]
        for (let c = 0; c < p; c++) {
          const tmp = other.get(0, c);
          for (let r = 0; r < m; r++) dst.set(r, c, this.get(r, 0) * tmp);
        }
      }

      return dst;
    } finally { Matrix.clearTempMatrices(); }
  }
}

// a column vector with direct contiguous storage
// this is just sugar so we can say e.g. new Vector(10) instead of new Matrix(10, 1)
export class Vector extends Matrix {

  // sugar over Matrix.getTempMatrix()
  static getTempVector(i, rows, data) { return Matrix.getTempMatrix(i, rows, 1, data); }

  // new Vector(rows) allocates a new Float64Array
  // new Vector(rows, array) uses the given array iff it's big enough
  constructor(rows, data) { super(rows, 1, data); }
}

// base class for other Matrix variants
class AbstractMatrix extends Matrix {

  init(rows, cols, data) {
    Matrix.checkDims(rows, cols);
    this.rows = rows;
    this.cols = cols;
    this.data = data;
    return this;
  }

  isColumnMajor()     { throw new Error(`isColumnMajor() not implemented in ${this.constructor.name}`); }
  isFloat64()         { throw new Error(`isFloat64() not implemented in ${this.constructor.name}`); }
  isDirect()          { throw new Error(`isDirect() not implemented in ${this.constructor.name}`); }
  isContiguous()      { throw new Error(`isContiguous() not implemented in ${this.constructor.name}`); }
  getData()           { throw new Error(`getData() not implemented in ${this.constructor.name}`); }
  reshape(rows, cols) { throw new Error(`reshape() not implemented in ${this.constructor.name}`); }
  reallocate(size)    { throw new Error(`reallocate() not implemented in ${this.constructor.name}`); }
  ensureStorage(size) { throw new Error(`ensureStorage() not implemented in ${this.constructor.name}`); }
  grow()              { throw new Error(`grow() not implemented in ${this.constructor.name}`); }
  shrink()            { throw new Error(`shrink() not implemented in ${this.constructor.name}`); }

  get(row, col)       { throw new Error(`get() not implemented in ${this.constructor.name}`); }
  set(row, col, val)  { throw new Error(`set() not implemented in ${this.constructor.name}`); }

  at(index)       { return this.get(index % this.rows, index / this.rows); }
  put(index, val) { return this.set(index % this.rows, index / this.rows, val); }
}

// a matrix with direct row major storage
// a main use for this is to allow more readable literals like
// [[1, 2],
//  [3, 4]]
class RowMatrix extends AbstractMatrix {

  init(rows, cols, data) {
    Matrix.checkDims(rows, cols);
    this.rows = rows;
    this.cols = cols;
    if (!isAnyArray(data)) {
      throw new RangeError(`cannot use ${data} as storage for ${this.dims} matrix: not an array`);
    }
    if (data.length !== rows) {
      throw new RangeError(`cannot use ${data} as storage for ${this.dims} matrix: ` +
                           `expected ${rows} rows, got ${data.length}`);
    }
    for (let r = 0; r < rows; r++) {
      if (!isAnyArray(data[r]) || data[r].length !== cols) {
        throw new RangeError(`cannot use ${data} as storage for ${this.dims} matrix: ` +
                             `row ${r} does not have length ${cols}`);
      }
    }
    this.data = data;
    this.allFloat64 = data.every(row => (row instanceof Float64Array));
    return this;
  }

  isColumnMajor() { return false; }
  isFloat64()     { return this.allFloat64; }
  isDirect()      { return true; }
  isContiguous()  { return false; }
  getData()       { return this.data; }

  get(row, col)      { return this.data[row][col]; }
  set(row, col, val) {        this.data[row][col] = val; return this; }
}

// base class for wrapped matrices
class WrappedMatrix extends AbstractMatrix {
  isColumnMajor() { return this.data.isColumnMajor(); }
  isFloat64()     { return this.data.isFloat64(); }
  isDirect()      { return false; }
  isContiguous()  { return false; }
  getData()       { return this.data.getData(); }
}

// implements Matrix.slice()
class SlicedMatrix extends WrappedMatrix {
  constructor(startRow, numRows, startCol, numCols, data) {
    super(numRows, numCols, data);
    reset(startRow, numRows, startCol, numCols, data);
  }
  reset(startRow, numRows, startCol, numCols, data) {
    init(numRows, numCols, data);
    if (startRow + numRows > data.rows || startCol + numCols > data.cols) {
      throw new RangeError
      (`cannot slice ${Matrix.dims(numRows, numCols)} at (${startRow}, ${startCol}) in ${data.dims}`);
    }
    this.startRow = startRow;
    this.startCol = startCol;
  }
  get(row, col)      { return this.data.get(this.startRow + row, this.startCol + col); }
  set(row, col, val) { return this.data.set(this.startRow + row, this.startCol + col, val); }
}

// implements Matrix.select*()
class SelectedMatrix extends WrappedMatrix {

  static checkIndices(indices, size, what) {
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      if (!Number.isInteger(index) || index < 0 || index >= size) {
        throw new RangeError(`${what} index ${i} invalid: ${index} not in range [0, ${size})`);
      }
    }
  }
  
  constructor(rowIndices, colIndices, data) {
    super(rowIndices.length, colIndices.length, data);
    reset(rowIndices, colIndices, data);
  }

  reset(rowIndices, colIndices, data) {
    init(rowIndices.length, colIndices.length, data);
    SelectedMatrix.checkIndices(rowIndices, data.rows, "row");
    SelectedMatrix.checkIndices(colIndices, data.cols, "col");
    this.rowIndices = rowIndices;
    this.colIndices = colIndices;
    return this;
  }

  get(row, col)      { return this.data.get(this.rowIndices[row], this.colIndices[col]); }
  set(row, col, val) { return this.data.set(this.rowIndices[row], this.colIndices[col], val); }
}

// implements Matrix.reshaped()
class ReshapedMatrix extends WrappedMatrix {
  init(rows, cols, data) {
    Matrix.checkDims(rows, cols);
    if (rows * cols > data.length) {
      throw new RangeError(`cannot reshape matrix with ${data.length} entries as ${this.dims}`);
    }
    this.rows = rows;
    this.cols = cols;
    this.data = data;
    return this;
  }
  isContiguous()     { return this.data.isContiguous(); }
  get(row, col)      { return this.data.at (col + this.rows + row); }
  set(row, col, val) { return this.data.put(col + this.rows + row, val); }
}

// common parts of implementation of Matrix.stacked() and Matrix.glued()
class BlockMatrix extends AbstractMatrix {

  static getRows(mats) {
    if (mats.length === 0) throw new Error("no matrices provided");
    const rows = mats[0].rows;
    for (let i = 1; i < mats.length; i++) {
      if (mats[i].rows !== rows) throw new RangeError(`matrix ${i} is ${mats[i].rows}x${mats[i].cols}, not ${rows}x?`);
    }
    return rows;
  }

  static getCols(mats) {
    if (mats.length === 0) throw new Error("no matrices provided");
    const cols = mats[0].cols;
    for (let i = 1; i < mats.length; i++) {
      if (mats[i].cols !== cols) throw new RangeError(`matrix ${i} is ${mats[i].rows}x${mats[i].cols}, not ?x${cols}`);
    }
    return cols;
  }

  static sumRows(mats) {
    if (mats.length === 0) throw new Error("no matrices provided");
    return mats.reduce((sum, m) => sum + m.rows, 0);
  }

  static sumCols(mats) {
    if (mats.length === 0) throw new Error("no matrices provided");
    return mats.reduce((sum, m) => sum + m.cols, 0);
  }

  constructor(rows, cols, data) {
    super(rows, cols, data);
    this.allColumnMajor = data.every(m => m.isColumnMajor());
    this.allFloat64 = data.every(m => m.isFloat64());
  }

  isColumnMajor() { return this.allColumnMajor; }
  isFloat64()     { return this.allFloat64; }
  isDirect()      { return false; }
  isContiguous()  { return false; }
}

// implements Matrix.stacked()
class StackedMatrix extends BlockMatrix {
  constructor(matrices) {
    super(BlockMatrix.sumRows(matrices), BlockMatrix.getCols(matrices), matrices);
    this.block = new Array(this.rows);
    let r = 0, o = 0;
    for (let i = 0; i < matrices.length; i++) {
      const mat = matrices[i];
      for (let j = 0; j < mat.rows; j++, r++) this.block[r] = { matrix: mat, offset: o };
      o += mat.rows;
    }
  }
  get(row, col) { return this.block[row].matrix.get(row - this.block[row].offset, col); }
  set(row, col) {        this.block[row].matrix.set(row - this.block[row].offset, col, val); return this; }
}

// implements Matrix.glued()
class GluedMatrix extends BlockMatrix {
  constructor(matrices) {
    super(BlockMatrix.getRows(matrices), BlockMatrix.sumCols(matrices), matrices);
    this.block = new Array(this.cols);
    let c = 0, o = 0;
    for (let i = 0; i < matrices.length; i++) {
      const mat = matrices[i];
      for (let j = 0; j < mat.cols; j++, c++) this.block[c] = { matrix: mat, offset: o };
      o += mat.cols;
    }
  }
  get(row, col) { return this.block[row].matrix.get(row, col - this.block[col].offset); }
  set(row, col) {        this.block[row].matrix.set(row, col - this.block[col].offset, val); return this; }
}
