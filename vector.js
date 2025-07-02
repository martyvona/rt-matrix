import { AbstractMatrix } from 'ml-matrix';
import { MatrixSubView } from 'ml-matrix';
import { isAnyArray } from 'is-any-array';

//a column vector either with internal storage as a single Float64Array() or that wraps another Matrix or array
//ml-matrix Matrix uses one Float64Array per row, so new Matrix(n, 1) does not have an ideal memory layout
//wrap(new Float64Array(n), { rows = n }) would work but does a little extra math in set(row, col) and get(row, col)
//this class also has some additional conveniences
export class Vector extends AbstractMatrix {

  //if v is null or undefined
  //* if length is not null: return a newly allocated Vector of the given length
  //* if length is null: throw an Error
  //if v is a Vector
  //* if length is not null: return v iff it has the given length, else throw an Error
  //* if length is null: return v
  //if v is a Matrix
  //* if neither v.columns nor v.rows is 1: throw an Error
  //* if length is not null: return Vector.wrap(v) iff v has the given length, else throw an Error
  //* if length is null: return Vector.wrap(v)
  //if v is any javascript array
  //* if length is not null: return Vector.wrap(v) iff v has the given length, else throw an Error
  //* if length is null: return Vector.wrap(v)
  static asVector(v, length = null) {
    if (v === null || v === undefined) {
      if (length !== null) return new Vector(length);
      else throw new Error("null argument");
    } else if (v instanceof Vector) {
      if (length === null || v.length === length) return v;
      else throw new Error(`expected Vector({length}), got Vector({v.length})`);
    } else if (Matrix.isMatrix(v)) {
      if (v.columns === 1 && (length === null || v.rows === length)) return Vector.wrap(v);
      else if (v.rows === 1 && (length === null || v.columns === length)) return Vector.wrap(v);
      else throw new Error(`cannot convert ${v.rows}x${v.columns} Matrix to Vector`);
    } else if (isAnyArray(v)) {
      if (length === null || v.length === length) return Vector.wrap(v);
      else throw new Error(`cannot wrap {v.length} array as Vector({length})`);
    }
    throw new Error(`cannot convert ${v} to Vector`);
  }

  //get the length of a Vector, column Matrix (nx1), row Matrix (1xn), or any javascript array
  static lengthOf(arrayOrMatrix) {
    if (arrayOrMatrix instanceof Vector) return arrayOrMatrix.length;
    else if (Matrix.isMatrix(arrayOrMatrix)) {
      if (arrayOrMatrix.columns === 1) return arrayOrMatrix.rows;
      else if (arrayOrMatrix.rows === 1) return arrayOrMatrix.columns;
      else throw new Error(`cannot get length of ${v.rows}x${v.columns} Matrix`);
    } else if (isAnyArray(arrayOrMatrix)) return arrayOrMatrix.length;
    else throw new Error(`cannot get length of {arrayOrMatrix}`);
  }

  //wrap a vector, a matrix with either 1 column or 1 row, or any javascript array as a Vector
  //the returned Vector reads and writes through to the passed object
  static wrap(arrayOrMatrix) { return new VectorWrapper(arrayOrMatrix); }

  //wrap part of a vector, a matrix with either 1 column or 1 row, or any javascript array as a Vector 
  //the returned Vector reads and writes through to the passed object
  static sub(arrayOrMatrix, start, length) { return new SubVector(arrayOrMatrix, start, length); }

  //wrap a selection of of a vector, a matrix with either 1 column or 1 row, or any javascript array as a Vector
  //the returned Vector reads and writes through to the passed object
  //the returned Vector at index i corresponds to the passed object at index redirect[i]
  static select(arrayOrMatrix, redirect) { return new VectorSelection(arrayOrMatrix, redirect); }

  //if arg is an integer:
  //* return a new Vector(arg) with preallocated storage as a Float64Array(arg) if arg >= 0
  //* return a new vector(-arg - 1) with no preallocated storage if arg < 0
  //if arg is a Vector: return a new Vector(arg.length) with data copied from arg
  //if arg is a Matrix
  //* if arg.columns === 1 (column vector): return a new Vector(arg.rows) with data copied from arg
  //* if arg.rows === 1 (row vector): return a new Vector(arg.columns) with data copied from arg
  //* otherwise throw an error
  //if arg is any javascript array: return a new Vector(arg.length) with data copied from arg
  constructor(arg) {
    super();
    let length = 0, getter = null;
    if (Number.isInteger(arg)) length = arg;
    else if (arg instanceof Vector) { length = arg.length; getter = i => arg.at(i); }
    else if (Matrix.isMatrix(arg)) {
      if (arg.columns === 1)   { length = arg.rows;        getter = i => arg.get(i, 0); }
      else if (arg.rows === 1) { length = arg.columns;     getter = i => arg.get(0, i); }
      else throw new Error(`cannot convert ${v.rows}x${v.columns} Matrix to Vector`);
    } else if (isAnyArray(arg)) { length = arg.length; getter = i => arg[i]; }
    this.columns = 1;
    this.rows = length >= 0 ? length : -length - 1;
    this.data = length >= 0 ? new Float64Array(length) : null;
    if (getter !== null) for (const i = 0; i < length; i++) this.data[i] = getter(i);
  }

  get length() { return rows; }

  //overrides AbstractMatrix.clone()
  clone() { return new Vector(this); }

  //overrides AbstractMatrix.set()
  set(row, col, value) {
    if (col === 0) this.data[row] = value;
    else throw new Error(`invalid column {col}`);
    return this;
  }

  //overrides AbstractMatrix.get()
  get(row, col) {
    if (col === 0 || col === undefined) return this.data[row];
    else throw new Error(`invalid column {col}`);
  }

  //like get() but doesn't need to validate col argument
  at(i) { return this.data[i]; }

  //like set() but doesn't need to validate col argument
  put(i, value) { this.data[i] = value; return this; }

  get x() { return at(0); }
  get y() { return at(1); }
  get z() { return at(2); }

  set x(v) { put(0, v); }
  set y(v) { put(1, v); }
  set z(v) { put(2, v); }

  withXY(arrayOrMatrix, func) {
    if (arrayOrMatrix instanceof Vector) { func(arrayOrMatrix.x, arrayOrMatrix.y); }
    else if (Matrix.isMatrix(arrayOrMatrix)) {
      const m = arrayOrMatrix;
      if      (m.rows === 2 && m.columns == 1) { func(m.get(0, 0), m.get(1, 0)); }
      else if (m.rows === 1 && m.columns == 2) { func(m.get(0, 0), m.get(0, 1)); }
      else throw new Error(`cannot get XY from {m.rows}x{m.columns} Matrix`);
    } else if (isAnyArray(arrayOrMatrix) && arrayOrMatrix.length == 2) {
      func(arrayOrMatrix[0], arrayOrMatrix[1]);
    } else throw new Error(`cannot get XY from {arrayOrMatrix}`);
  }

  putXY(x, y, dest) {
    if (!isAnyArray(dest)) { return Vector.asVector(dest, 3).put(0, x).put(1, y); }
    else if (dest.length === 2) { dest[0] = x; dest[1] = y; return dest; } //avoid creating wrapper
    else throw new Error(`cannot write XY to {dest}`);
  }

  withXYZ(arrayOrMatrix, func) {
    if (arrayOrMatrix instanceof Vector) { return func(arrayOrMatrix.x, arrayOrMatrix.y, arrayOrMatrix.z); }
    else if (Matrix.isMatrix(arrayOrMatrix)) {
      const m = arrayOrMatrix;
      if      (m.rows === 3 && m.columns == 1) { return func(m.get(0, 0), m.get(1, 0), m.get(2, 0)); }
      else if (m.rows === 1 && m.columns == 3) { return func(m.get(0, 0), m.get(0, 1), m.get(0, 2)); }
      else throw new Error(`cannot get XYZ from {m.rows}x{m.columns} Matrix`);
    } else if (isAnyArray(arrayOrMatrix) && arrayOrMatrix.length == 3) {
      return func(arrayOrMatrix[0], arrayOrMatrix[1], arrayOrMatrix[2]);
    } else throw new Error(`cannot get XYZ from {arrayOrMatrix}`);
  }

  putXYZ(x, y, z, dest) {
    if (!isAnyArray(dest)) { return Vector.asVector(dest, 3).put(0, x).put(1, y).put(2, z); }
    else if (dest.length === 3) { dest[0] = x; dest[1] = y; dest[2] = z; return dest; } //avoid creating wrapper
    else throw new Error(`cannot write XYZ to {dest}`);
  }

  //dest = this x arrayOrMatrix
  cross(otherArrayOrMatrix, dest = this) {
    if (this.length != 3) throw new Error("cross product requires Vector(3)");
    const tx = this.x, ty = this.y, tz = this.z;
    return withXYZ(otherArrayOrMatrix, (ox, oy, oz) => {
      //tx ty tz
      //ox oy oz
      const x = ty * oz - tz * oy;
      const y = tz * ox - tx * oz;
      const z = tx * oy - ty * ox;
      return putXYZ(x, y, z, dest);
    });
  }

  crossProductMatrix(dest = null) { return crossJacobianWRTOther(dest); }

  crossJacobianWRTOther(dest = null, dr = 0, dc = 0) {
    // 0, -z,  y
    // z,  0, -x
    //-y,  x,  0
    dest = dest || new Matrix(3, 3);
    return withXYZ(this, (x, y, z) => {
      dest.set(dr + 1, dc + 0,  z); dest.set(dr + 0, dc + 1, -z); dest.set(dr + 0, dc + 2,  y);
      dest.set(dr + 2, dc + 0, -y); dest.set(dr + 2, dc + 1,  x); dest.set(dr + 1, dc + 2, -x);
      return dest;
    });
  }

  crossJacobianWRTThis(otherArrayOrMatrix, dest = null, dr = 0, dc = 0) {
    // 0,  z, -y
    //-z,  0,  x
    // y, -x,  0
    dest = dest || new Matrix(3, 3);
    return withXYZ(otherArrayOrMatrix, (x, y, z) => {
      dest.set(dr + 1, dc + 0, -z); dest.set(dr + 0, dc + 1,  z); dest.set(dr + 0, dc + 2, -y);
      dest.set(dr + 2, dc + 0,  y); dest.set(dr + 2, dc + 1, -x); dest.set(dr + 1, dc + 2,  x);
      return dest;
    });
  }

  crossJacobian(dest = null) { return crossJacobianWRTThis(crossJacobianWRTOther(dest || new Matrix(3, 6), 0, 3)); }

  //view this Vector as the columns of a Matrix
  //the returned Matrix reads and writes through to this Vector
  //the length of this Vector must be evenly divisible by rows
  //the returned Matrix at row i column j corresponds to this Vector at j * rows + i
  asMatrix(rows) { return new MatrixWrapper(this, rows); }

  //TODO negate(dest = this)
  //TODO add(otherNumberOrArrayOrMatrix, dest = this)
  //TODO sub(otherNumberOrArrayOrMatrix, dest = this)
  //TODO dot(otherOrArrayOrMatrix)
  //TODO scale(number, dest = this)
  //TODO preMul(matrix, dest = this)
  //TODO fast mmul
}

export class MatrixWrapper extends AbstractMatrix {
  constructor(vector, rows) {
    if (rows <= 0 || vector % rows != 0) throw new Error(`cannot wrap {vector} as a Matrix with {rows} rows`)
    this.data = vector;
    this.rows = rows;
    this.columns = v.length / rows;
  }
  set(row, col, value) { return data.put(col * this.rows + row, value); }
  get(row, col) { return data.at(col * this.rows + row); }
}

class VectorWrapperBase extends Vector {

  constructor(length, arrayOrMatrix) {
    super(-length);
    this.data = arrayOrMatrix;
    if (arrayOrMatrix instanceof Vector) {
        this.setter = (i, value) => this.data.put(i, value);
        this.getter = i => this.data.at(i);
    } else if (Matrix.isMatrix(arrayOrMatrix)) {
      if (matrix.rows === 1) {
        this.setter = (i, value) => this.data.set(i, 0, value);
        this.getter = i => this.data.get(i, 0);
      } else if (matrix.columns === 1) {
        this.setter = (i, value) => this.data.set(0, i, value);
        this.getter = i => this.data.get(0, i);
      } else throw new Error(`cannot wrap {arrayOrMatrix} as Vector`);
    } else if (isAnyArray(arrayOrMatrix)) {
        this.setter = (i, value) => this.data[i] = value;
        this.getter = i => this.data[i];
    } else throw new Error(`cannot wrap {arrayOrMatrix} as Vector`);
  }

  index(i) { return i; }

  set(row, col, value) {
    if (col === 0) return this.setter(row, value);
    else throw new Error(`invalid column {col}`);
  }

  get(row, col) {
    if (col === 0) return this.getter(row);
    else throw new Error(`invalid column {col}`);
  }

  at(i) { return this.getter(i); }

  put(i, value) { return this.setter(i, value); }
}

export class VectorWrapper extends VectorWrapperBase {
  constructor(arrayOrMatrix) { super(Vector.lengthOf(arrayOrMatrix), arrayOrMatrix); }
}

export class SubVector extends VectorWrapperBase {
  constructor(arrayOrMatrix, start, length) { super(length, arrayOrMatrix); this.start = start; }
  index(i) {
    if (i < 0 || i >= this.length) throw new RangeError(`index {i} out of range [0,{length}-1]`);
    return i - this.start;
  }
}

export class VectorSelection extends VectorWrapperBase {
  constructor(arrayOrMatrix, redirect) { super(redirect.length, arrayOrMatrix); this.redirect = redirect; }
  index(i) {
    if (i < 0 || i >= this.length) throw new RangeError(`index {i} out of range [0, {this.length}-1]`);
    return this.redirect[i];
  }
}

export class Vector2 extends Vector {

  static asVector2(v) {
    if (v === null || v === undefined) return new Vector2();
    else if (v instanceof Vector) {
      if (v.length === 2) return v;
      else throw new Error(`expected Vector(2), got Vector({v.length})`);
    } else if (Matrix.isMatrix(v)) {
      if (v.columns === 1 && v.rows === 2) return Vector.wrap(v);
      else if (v.rows === 1 && v.columns === 2) return Vector.wrap(v);
      else throw new Error(`cannot convert ${v.rows}x${v.columns} Matrix to Vector2`);
    } else if (isAnyArray(v)) {
      if (v.length === 2) return Vector.wrap(v);
      else throw new Error(`cannot wrap {v.length} array as Vector2`);
    }
    throw new Error(`cannot convert ${v} to Vector2`);
  }

  constructor(x, y) {
    if (x === undefined) super(2);
    else if (y === undefined) {
      if (Vector.lengthOf(x) != 2) throw new Error(`cannot convert {x} to Vector2`);
      super(x);
    } else super(2);
    this.x = x;
    this.y = y;
  }

  get length() { return 2; }

  clone() { return new Vector2(this); }
}

export class Vector3 extends Vector {

  static asVector3(v) {
    if (v === null || v === undefined) return new Vector3();
    else if (v instanceof Vector) {
      if (v.length === 3) return v;
      else throw new Error(`expected Vector(3), got Vector({v.length})`);
    } else if (Matrix.isMatrix(v)) {
      if (v.columns === 1 && v.rows === 3) return Vector.wrap(v);
      else if (v.rows === 1 && v.columns === 3) return Vector.wrap(v);
      else throw new Error(`cannot convert ${v.rows}x${v.columns} Matrix to Vector3`);
    } else if (isAnyArray(v)) {
      if (v.length === 3) return Vector.wrap(v);
      else throw new Error(`cannot wrap {v.length} array as Vector3`);
    }
    throw new Error(`cannot convert ${v} to Vector3`);
  }

  constructor(x, y, z) {
    if (x === undefined) super(3);
    else if (y === undefined) {
      if (Vector.lengthOf(x) != 3) throw new Error(`cannot convert {x} to Vector3`);
      super(x);
    } else super(3);
    this.x = x;
    this.y = y;
    this.z = z;
  }

  get length() { return 3; }

  clone() { return new Vector3(this); }
}
