import React from 'react';

const Que5 = () => {
  const findUnion = (arr1, arr2) => {
    return [...new Set([...arr1, ...arr2])];
  };

  const arr1 = [1, 2, 3, 4];
  const arr2 = [3, 4, 5, 6];
  return <p>Union: {JSON.stringify(findUnion(arr1, arr2))}</p>;
};

export default Que5;
