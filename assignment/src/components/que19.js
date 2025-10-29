import React from 'react';

const Que19 = () => {
  const objectToArray = (obj) => {
    return Object.entries(obj);
  };

  const obj = { name: 'John', age: 25 };
  return <p>Key-Value Pairs: {JSON.stringify(objectToArray(obj))}</p>;
};

export default Que19;
