import React from 'react';

const Que6 = () => {
  const bubbleSort = (arr) => {
    let swapped;
    do {
      swapped = false;
      for (let i = 0; i < arr.length - 1; i++) {
        if (arr[i] > arr[i + 1]) {
          [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]];
          swapped = true;
        }
      }
    } while (swapped);
    return arr;
  };

  const arr = [5, 3, 8, 1, 2];
  return <p>Sorted Array: {JSON.stringify(bubbleSort(arr))}</p>;
};

export default Que6;
