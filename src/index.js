const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');
const { normalize } = require('./utils');

const filePath = './data/data.csv';
const features = [];
const labels = [];
const locationMap = {};

// Load CSV
fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (row) => {
        if (!locationMap[row.location]) {
            locationMap[row.location] = Object.keys(locationMap).length + 1;
        }
        const encodedLocation = locationMap[row.location];
        features.push([parseFloat(row.square_feet), parseFloat(row.bedrooms), parseFloat(row.bathrooms), encodedLocation]);
        labels.push([parseFloat(row.price)]);
    })
    .on('end', () => {
        const featureTensor = tf.tensor2d(features);
        const labelTensor = tf.tensor2d(labels);

        const normalizedFeatures = normalize(featureTensor);
        const normalizedLabels = normalize(labelTensor);

        trainModel(normalizedFeatures, normalizedLabels);
    });

async function trainModel(features, labels) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [features.shape[1]] }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    console.log('Training...');
    await model.fit(features, labels, { epochs: 500, batchSize: 32 });
    await model.save('file://./models/house-price-model');
    console.log('Model trained and saved.');
}
