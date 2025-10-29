import React from 'react';

const Que14 = () => {
  const mergeByKey = (arr1, arr2, key) => {
    return arr1.map(item1 => ({
      ...item1,
      ...(arr2.find(item2 => item2[key] === item1[key]) || {})
    }));
  };

  const arr1 = [{ id: 1, name: 'John' }, { id: 2, name: 'Alice' }];
  const arr2 = [{ id: 1, age: 25 }, { id: 2, age: 30 }];
  return <p>Merged Array: {JSON.stringify(mergeByKey(arr1, arr2, 'id'))}</p>;
};

export default Que14;
