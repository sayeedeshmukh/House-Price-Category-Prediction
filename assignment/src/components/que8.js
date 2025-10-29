import React from 'react';
const Que8 = () => {
  const findMostFrequent = (arr) => {
    const freqMap = {};
    let maxCount = 0;
    let mostFrequent;
    arr.forEach(item => {
      freqMap[item] = (freqMap[item] || 0) + 1;
      if (freqMap[item] > maxCount) {
        maxCount = freqMap[item];
        mostFrequent = item;
      }
    });
    return mostFrequent;
  };
  const arr = [1, 2, 2, 3, 3, 3, 4];
  return <p>Most Frequent: {findMostFrequent(arr)}</p>;
};
export default Que8;
