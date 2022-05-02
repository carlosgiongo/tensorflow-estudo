const tf = require('@tensorflow/tfjs-node');

async function main (){
    const model = tf.sequential()
    model.add(tf.layers.dense({units:1, inputShape: [4]}))
    
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    })
    
    const xs = tf.tensor2d([ 
        [1, 0, 0, 1], 
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1] 
    ])  //8 coleções de 4 elementos
    const ys = tf.tensor2d([[2], [3], [1], [0], [4], [2], [2], [3]]) //8 coleções de 1 elementos
    
    await model.fit(xs, ys, {epochs: 750})
    
    let result = model.predict(tf.tensor2d([1, 1, 0, 1], [1, 4]))
    let dado = await result.dataSync()
    console.log("Resultado: " + Math.round(dado))
}

main()