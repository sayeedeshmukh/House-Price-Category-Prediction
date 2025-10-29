import React from 'react';

const Que17 = () => {
  const countFrequency = (arr, key) => {
    return arr.reduce((count, obj) => {
      count[obj[key]] = (count[obj[key]] || 0) + 1;
      return count;
    }, {});
  };

  const arr = [{ age: 25 }, { age: 30 }, { age: 25 }];
  return <p>Frequency of Ages: {JSON.stringify(countFrequency(arr, 'age'))}</p>;
};

export default Que17;
