const tf = require('@tensorflow/tfjs-node');

async function main (){
    let model = tf.sequential()
    model = await treinar(model, [[-1], [0], [1], [2], [3], [4]], [[-3], [-1], [1], [3], [5], [7]])
    let result = await predict([[20], [19]], model)
    console.log("=>", result)
}

async function predict(valor, model){
    let result = model.predict(tf.tensor2d(valor))
    let dado = await result.dataSync()
    return dado
}

async function treinar(model, inputs, outputs){
    model.add(tf.layers.dense({units:1, inputShape: [1]}))
    
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    })

    const xs = tf.tensor2d(inputs)  //8 coleções de 4 elementos
    const ys = tf.tensor2d(outputs) //8 coleções de 1 elementos
    await model.fit(xs, ys, {epochs: 750})
    return model
}

main()