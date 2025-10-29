import React from 'react';

const Que16 = () => {
  const convertToMap = (arr, key) => {
    return arr.reduce((map, obj) => {
      map[obj[key]] = obj;
      return map;
    }, {});
  };

  const arr = [{ id: 1, name: 'John' }, { id: 2, name: 'Alice' }];
  return <p>Map: {JSON.stringify(convertToMap(arr, 'id'))}</p>;
};

export default Que16;
