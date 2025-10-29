import React from 'react';

const Que1 = () => {
  const findSecondLargest = (arr) => {
    let first = -Infinity, second = -Infinity;
    for (let num of arr) {
      if (num > first) {
        second = first;
        first = num;
      } else if (num > second && num !== first) {
        second = num;
      }
    }
    return second;
  };

  const arr = [20, 30, 40, 50, 34, 44];
  return <p>Second Largest: {findSecondLargest(arr)}</p>;
};

export default Que1;
