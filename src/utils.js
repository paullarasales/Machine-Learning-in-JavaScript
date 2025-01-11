// Normalize tensor data
function normalize(tensor) {
    const min = tensor.min();
    const max = tensor.max();
    return tensor.sub(min).div(max.sub(min));
}

module.exports = { normalize };
