import React from 'react';

const Que3 = () => {
  const flatten = (arr) => {
    let result = [];
    for (let item of arr) {
      if (Array.isArray(item)) {
        result = result.concat(flatten(item));
      } else {
        result.push(item);
      }
    }
    return result;
  };

  const nestedArr = [1, [2, [3, [4, 5]]]];
  return <p>Flattened Array: {JSON.stringify(flatten(nestedArr))}</p>;
};

export default Que3;
