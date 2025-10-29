import React from 'react';

const Que9 = () => {
  const rotateArray = (arr, k) => {
    const length = arr.length;
    k = k % length;
    return [...arr.slice(length - k), ...arr.slice(0, length - k)];
  };

  const arr = [1, 2, 3, 4, 5];
  return <p>Rotated Array: {JSON.stringify(rotateArray(arr, 2))}</p>;
};

export default Que9;
