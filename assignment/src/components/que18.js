import React from 'react';

const Que18 = () => {
  const filterObjects = (arr, conditions) => {
    return arr.filter(item => {
      return Object.keys(conditions).every(key => item[key] === conditions[key]);
    });
  };

  const arr = [{ name: 'John', age: 25 }, { name: 'Alice', age: 30 }];
  const conditions = { age: 25 };
  return <p>Filtered Objects: {JSON.stringify(filterObjects(arr, conditions))}</p>;
};

export default Que18;
