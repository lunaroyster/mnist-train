const brain = require('brain.js');
const fse = require('fs-extra');

(async function() {
    
    let file  = await fse.readFile(`${__dirname}/data/mnist_train.csv`, 'utf8');
    let rows = file.split('\n');
    let tests = [];
    
    console.log("Compiling training data");
    for(let row of rows) {
        if(row.length==0) continue;
        let rowItems = row.split(',');
        let arr = [0,0,0,0,0,0,0,0,0,0];
        arr[rowItems[0]] = 1;
        tests.push(({
            input: rowItems.splice(1).map(m => m/256),
            output: arr
        }));
    }
    console.log(`Compiled training data: ${tests.length}`);
    console.log("Logging compiled training data");
    await fse.writeFile(`training.json`, tests);
    console.log("Logged compiled training data");
    var net = new brain.NeuralNetwork({activation: 'sigmoid', hiddenLayers: [15], learningRate: 0.8});
    console.log("Beginning training.");
    net.train(tests, {log: true, errorThresh: 0.006, logPeriod: 1});
    console.log("Trained.");
    await fse.writeFile(`net.json`, JSON.stringify(net.toJSON()));
})()
.catch(error=> {
    console.log(error);
});


