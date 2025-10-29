import React from 'react';

const Que7 = () => {
  const chunkArray = (arr, size) => {
    const chunks = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  };

  const arr = [1, 2, 3, 4, 5, 6];
  return <p>Chunks: {JSON.stringify(chunkArray(arr, 2))}</p>;
};

export default Que7;
