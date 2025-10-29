import React from 'react';

const Que11 = () => {
  const sortByKey = (arr, key) => {
    return arr.sort((a, b) => a[key] > b[key] ? 1 : -1);
  };

  const arr = [{ id: 3, name: 'John' }, { id: 1, name: 'Alice' }, { id: 2, name: 'Bob' }];
  return <p>Sorted by ID: {JSON.stringify(sortByKey(arr, 'id'))}</p>;
};

export default Que11;
