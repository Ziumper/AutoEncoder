import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by komoszet on 2017-05-04.
 */
public class SingleLayerMinstExample {

    private static Logger log = LoggerFactory.getLogger(SingleLayerMinstExample.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // ilość klas wyjściowych
        int batchSize = 128; // ilość danych testowych i treningowych
        int rngSeed = 123; // losowe ziarno dla inicjalizacji wag
        int numEpochs = 15; // ilosć iteracji dla danych

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility inicjalizacja wag
                // use stochastic gradient descent as an optimization algorithm optymalizacja funkcji
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)//ilość prównań podczas procesu uczenia, do minimalizacji błędu
                .learningRate(0.006) //specify the learning rate współczynnik uczenia sieci
                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate. współczynnik kierunku optrymalizacji (mówi jak szybko algorytm optymalizacyjny
                // będzie działał na loklane minimum i wagi
                .regularization(true).l2(1e-4) // zabeczpieczenie przed dopasowaniem
                .list() //metoda przełączająca na dodawanie kolejnych warstw
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns) //liczba punktów wejściowych piksele
                        .nOut(1000) //liczba punktów wyjściowych
                        .activation(Activation.RELU) //funkcja aktywacji
                        .weightInit(WeightInit.XAVIER) //sposob inicjalizacji wag
                        .build()) // metoda budująca
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");


//        Celność (Accuracy) - procent obrazków poprawnie sklasyfikowanych
//        Precyzja (Precision) - liczba poprawnie określonych przez ilosć niepoprawnie określonych The number of true positives divided by the number of true positives and false positives.
//        Odwołanie ( Recall ) - Liczba poprwanie  sklasyfikowanych podzielona przez sumę poprawnie sklasyfikowanych i źle sklasyfikowanych
//        F1 Score - średnia ważona pomiędzy precyzją i odwolaniem
    }
}
