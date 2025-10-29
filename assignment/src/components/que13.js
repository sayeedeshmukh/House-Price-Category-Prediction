import React from 'react';

const Que13 = () => {
  const findMaxByKey = (arr, key) => {
    return arr.reduce((max, obj) => (obj[key] > max[key] ? obj : max), arr[0]);
  };

  const arr = [{ id: 1, value: 10 }, { id: 2, value: 30 }, { id: 3, value: 20 }];
  return <p>Object with Maximum Value: {JSON.stringify(findMaxByKey(arr, 'value'))}</p>;
};

export default Que13;
