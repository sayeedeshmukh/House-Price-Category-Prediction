import React from 'react';

const Que20 = () => {
  const deepClone = (obj) => {
    return JSON.parse(JSON.stringify(obj));
  };

  const obj = { name: 'John', details: { age: 25, address: { city: 'NY' } } };
  return <p>Cloned Object: {JSON.stringify(deepClone(obj))}</p>;
};

export default Que20;
