const brain = require('brain.js');
const fse = require('fs-extra');

(async function() {
    
    let file  = await fse.readFile(`${__dirname}/data/mnist_test.csv`, 'utf8');
    let rows = file.split('\n');
    let tests = [];
    console.log("Compiling testing data");
    for(let row of rows) {
        if(row.length==0) continue;
        let rowItems = row.split(',');
        let arr =[0,0,0,0,0,0,0,0,0,0];
        arr[rowItems[0]] = 1;
        tests.push(({
            input: rowItems.splice(1).map(m => m/256),
            output: arr
        }));
    }
    console.log(`Compiled training data: ${tests.length} samples`);

    var net = new brain.NeuralNetwork({activation: 'sigmoid', hiddenLayers: [7], learningRate: 0.9});
    console.log("Loading NN config.");
    let NNConfig = await fse.readFile(`${__dirname}/net.json`);
    net.fromJSON(JSON.parse(NNConfig));
    
    let outcomes = [];
    let successes = 0;
    for(let test of tests) {
        let rawOutcome = net.run(test.input);
        let outcome = rawOutcome.indexOf(Math.max(...rawOutcome));
        let success = test.output.indexOf(1) == outcome;
        outcomes.push(success);
        if(success) successes++;
    }
    console.log(successes/outcomes.length);
    
})()
.catch(error=> {
    console.log(error);
});


