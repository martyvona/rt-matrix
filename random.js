//seedable random number generator
//https://stackoverflow.com/a/47593316
//https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random
export default function Random(seed = null) {

  function sfc32(a, b, c, d) {
    return function() {
      a |= 0; b |= 0; c |= 0; d |= 0;
      let t = (a + b | 0) + d | 0;
      d = d + 1 | 0;
      a = b ^ b >>> 9;
      b = c + (c << 3) | 0;
      c = (c << 21 | c >>> 11);
      c = c + t | 0;
      return (t >>> 0) / 4294967296;
    }
  }

  function cyrb128(str) {
    let h1 = 1779033703, h2 = 3144134277,
        h3 = 1013904242, h4 = 2773480762;
    for (let i = 0, k; i < str.length; i++) {
      k = str.charCodeAt(i);
      h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
      h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
      h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
      h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
    }
    h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
    h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
    h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
    h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
    h1 ^= (h2 ^ h3 ^ h4), h2 ^= h1, h3 ^= h1, h4 ^= h1;
    return [h1>>>0, h2>>>0, h3>>>0, h4>>>0];
  }

  if (seed === null || seed === undefined) seed = Date.now();
  seed = cyrb128(seed.toString());

  const next = sfc32(seed[0], seed[1], seed[2], seed[3]);
  const nextUnder = max => next() * max;
  const nextIn = (min, max) => next() * (max - min) + min;
  const nextInt = max => Math.floor(next() * Math.floor(max));
  const nextIntIn = (min, max) => {
    const mc = Math.ceil(min);
    return Math.floor(next() * (Math.floor(max) - mc) + mc);
  }
  const nextIntInclusive = (min, max) => {
    const mc = Math.ceil(min);
    Math.floor(next() * (Math.floor(max)- mc + 1) + mc);
  };

  return {
    next,            //[0.0, 1.0)
    nextUnder,       //[0.0, max)
    nextIn,          //[min, max)
    nextInt,         //[0, max)
    nextIntIn,       //[min, max)
    nextIntInclusive //[min, max]
  };
}
