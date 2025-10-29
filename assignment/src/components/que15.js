import React from 'react';

const Que15 = () => {
  const removeDuplicatesByKey = (arr, key) => {
    const seen = new Set();
    return arr.filter(item => {
      if (seen.has(item[key])) {
        return false;
      }
      seen.add(item[key]);
      return true;
    });
  };

  const arr = [{ id: 1, name: 'John' }, { id: 2, name: 'Alice' }, { id: 1, name: 'John' }];
  return <p>Without Duplicates: {JSON.stringify(removeDuplicatesByKey(arr, 'id'))}</p>;
};

export default Que15;
