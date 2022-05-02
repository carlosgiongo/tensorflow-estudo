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

async function treinar(model, inputs, outputs, shape = [1], epocas = 750, compilacao_confs = {loss: 'meanSquaredError', optimizer: 'sgd'}){
    model.add(tf.layers.dense({units:1, inputShape: shape}))
    model.compile(compilacao_confs)

    const xs = tf.tensor2d(inputs)  //x coleções de y elementos
    const ys = tf.tensor2d(outputs) //x coleções de y elementos
    await model.fit(xs, ys, {epochs: epocas})
    return model
}

main()