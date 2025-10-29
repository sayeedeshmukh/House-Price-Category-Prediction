import React from 'react';

const Que12 = () => {
  const groupBy = (arr, key) => {
    return arr.reduce((result, currentValue) => {
      (result[currentValue[key]] = result[currentValue[key]] || []).push(currentValue);
      return result;
    }, {});
  };

  const arr = [{ age: 25, name: 'John' }, { age: 30, name: 'Alice' }, { age: 25, name: 'Bob' }];
  return <p>Grouped by Age: {JSON.stringify(groupBy(arr, 'age'))}</p>;
};

export default Que12;
