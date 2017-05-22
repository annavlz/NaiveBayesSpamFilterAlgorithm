fs = require("fs")
R = require("ramda")

var trainFileName, testFileName
[trainFileName, testFileName] = process.argv.slice(2)

const parseFile = function (rawString) {
    let lines = R.pipe(
        R.split("\n"),
        R.map(R.pipe(
            R.split("     "),
            R.reject(R.isEmpty),
            R.map((char) => parseInt(char))
        ))
    )(rawString)
    return R.reject(R.isEmpty, lines)
}

const fillTable = (table, row) => {
    let [features, label] = R.splitAt(-1, row)
    if(label == 1){
        return {
            notSpam: table.notSpam,
            spam: R.zipWith(R.add, table.spam, features),
            totalSpam: R.inc(table.totalSpam),
            totalNotSpam: table.totalNotSpam
        }
    }else{
        return {
            spam: table.spam,
            notSpam: R.zipWith(R.add, table.notSpam, features),
            totalSpam: table.totalSpam,
            totalNotSpam:  R.inc(table.totalNotSpam)
        }
    }
}

const getModel = (data) => {
    let table = R.reduce(fillTable, {spam: R.repeat(0, 12), notSpam: R.repeat(0, 12), totalSpam: 0, totalNotSpam: 0}, data)
    return {
        spamFeatures: table.spam,
        notSpamFeatures: table.notSpam,
        numberOfSpams: table.totalSpam,
        numberOfNotSpams: table.totalNotSpam,
        frequency: R.zipWith(R.add, table.spam, table.notSpam),
        primeSpam: table.totalSpam/R.length(data),
        primeNotSpam: table.totalNotSpam/R.length(data)
    }
}

const caclProbs = (testCase, features, total) => {
    return R.zipWith((c, p) => {
        if(c > 0){
            return p/total
        }else{
            return 1 - (p/total)
        }
    }, testCase, features)
}

const getProbabilities = (testCase, model, type) => {
    if(type == "spam"){
        return caclProbs(testCase, model.spamFeatures, model.numberOfSpams) 
    }else{
        return caclProbs(testCase, model.notSpamFeatures, model.numberOfNotSpams)
    }
}

const toPercent = (number) => {
    return `${Math.round(number * 100)}%`
}

const predictLabels = (data, testData) => {
    let model = getModel(data)
    return R.map((testCase) => {
        let totalNumber = model.numberOfNotSpams + model.numberOfSpams
        let pSpam = R.reduce(R.multiply, 1, getProbabilities(testCase, model, "spam"))
        let pNotSpam = R.reduce(R.multiply, 1, getProbabilities(testCase, model, "notSpam"))
        let denominator = pSpam * model.primeSpam + pNotSpam * model.primeNotSpam
        let mightBeSpam = (pSpam * model.primeSpam) / denominator
        let mightBeNotSpam = (pNotSpam * model.primeNotSpam) / denominator
        let label = (mightBeNotSpam > mightBeSpam) ? " Not spam" : " Spam    "
        return `${testCase.join(",")} ${label}      ${toPercent(mightBeSpam)}    ${toPercent(mightBeNotSpam)}` 
    }, testData)
}


const trainFileRaw = parseFile(fs.readFileSync(trainFileName).toString())
const testFileRaw = parseFile(fs.readFileSync(testFileName).toString())
let result = predictLabels(trainFileRaw, testFileRaw).join("\n")
let heading = "Features                 Label        Spam   Not spam\n"
fs.writeFile("sampleoutput.txt", heading + result, function(err) {
    if(err) {
        return console.log(err);
    }

    console.log("The file was saved!");
}); 