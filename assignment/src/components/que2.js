import React from 'react';

const Que2 = () => {
  const removeDuplicates = (arr) => {
    const unique = [];
    for (let item of arr) {
      if (!unique.includes(item)) {
        unique.push(item);
      }
    }
    return unique;
  };

  const arr = [1, 2, 2, 3, 4, 4, 5];
  return <p>Unique Array: {JSON.stringify(removeDuplicates(arr))}</p>;
};

export default Que2;
