import React from 'react';

const Que10 = () => {
  const findMissing = (arr, start, end) => {
    const missing = [];
    for (let i = start; i <= end; i++) {
      if (!arr.includes(i)) missing.push(i);
    }
    return missing;
  };

  const arr = [1, 2, 4, 6, 7];
  return <p>Missing Numbers: {JSON.stringify(findMissing(arr, 1, 7))}</p>;
};

export default Que10;
