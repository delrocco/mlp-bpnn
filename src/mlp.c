//==============================================================================
// Joseph Del Rocco
// EEL 6812
// Spring 2011
// Final Project: MLP BPNN Experiements w/ Back Propagation Variations
//==============================================================================
// ansi-c
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>
// unix
#ifndef WIN32
#include <sys/wait.h>
#endif

//-------------------------------------
// CONSTANTS, MACROS, DEFS
//-------------------------------------

#define uchar        unsigned char
#define ushort       unsigned short
#define uint         unsigned int
#define true         (1)
#define false        (0)
#ifdef WIN32
#pragma warning      (disable:4267)
#pragma warning      (disable:4018)
#else
#define min(a,b)     ((a<b)?a:b)
#define max(a,b)     ((a>b)?a:b)
#endif
#define sign(a)      (((a)>0.0)?1.0:(((a)<0.0)?-1.0:0.0))
#define fixzero(a)   (a = ((a==0 && (((unsigned char*)(&a))[sizeof(a)-1] & 0x80))?0:a))

#define PATTERN(p)   g_dataset.patterns[g_dataset.order[p]]  // <-- USE THIS!!
#define LEARNING(a)  (g_learning.method == a)
#define O_LAYER      g_model.numLayers-1
#define I_LAYER      0
#define O_COUNT      g_model.numONodes
#define H_COUNT      g_model.numHNodes
#define I_COUNT      g_model.numINodes
#define DELIMITERS   ",\r\n\0\t"
#define CONVG_1_LIM  (0.6) // > this = node converged to 'true'
#define CONVG_0_LIM  (0.4) // < this = node converged to 'false'
#define CONVG_ERROR  (0.1) // default

#define CMD_CREATE   (1<<0)
#define CMD_TRAIN    (1<<1)
#define CMD_TEST     (1<<2)
#define CMD_VIEW     (1<<3)
#define CMD_NMN      (1<<4)
#define ADDITIVE     (1<<27)
#define USE_ERROR    (1<<28)
#define SHOW_LEARN   (1<<29)
#define SHOW_MODEL   (1<<30)
#define XOR_MODEL    (1<<31)

enum LearningMethods
{
  LEARN_GD=0,            // Gradient Descent
  LEARN_GDM,             // Gradient Descent w/ Momentum
  LEARN_RPROP,           // Resilient Propagation (RPROP)
  MAX_LEARN_METHODS
};

typedef struct _Pattern
{
  double*   inputs;      // inputs per pattern
  uint      output;      // expected output
} Pattern;

typedef struct _DataSet
{
  char*     filename;    // filename of train/test data to load
  uchar     dimInput;    // dimensionality of input
  uchar     dimOutput;   // dimensionality of output
  uint      numPatterns; // total # of patterns (PT)
  uint*     order;       // pattern selection order
  Pattern*  patterns;    // list of patterns
} DataSet;

typedef struct _Learning
{
  uchar     method;      // type of BP learning method
  uint      epochs;      // max epochs of training
  double    eta;         // learning rate
  double    alpha;       // gradient decent momentum
  uint      correct;     // number of correct classifications
  double    cnvgerr;     // classification error threshold for convergence, if used
  double    terror;      // total error using avg. sum of squared errors
} Learning;

typedef struct _NNWeight
{
  double    w;           // weight value
  double    grad;        // gradient of error function
  double    gradLast;    // gradient of error function (last epoch)
  double    dw;          // delta weight change
  double    dwLast;      // delta weight change (last epoch)
  double    sz;          // RPROP update value
  double    szLast;      // RPROP update value (last epoch)
} NNWeight;

typedef struct _NNNode
{
  double    output;      // output = g(net input)
  double    delta;       // delta error term
  NNWeight* weights;     // weights eminating from each node
} NNNode;

typedef struct _NNModel
{
  char*     filename;    // model name (and file)
  uint      seed;        // randomizing seed (stored w/ model)
  uchar     numLayers;   // total # of layers including input and output
  uchar     numINodes;   // dimensionality of input + bias
  uchar     numHNodes;   // dimensionality of hidden layer + bias
  uchar     numONodes;   // dimensionality of output
  double    error;       // classification error
  double    best;        // classification error (best of all time)
  NNNode**  nodes;       // neurons
} NNModel;

//-------------------------------------
// PROTOTYPES
//-------------------------------------

int    initialize(int argc, char** argv);
void   cleanup();
int    createModel();
int    trainModel();
int    testModel();
int    loadModel();
int    loadDataSet();
int    saveModel();
uint   nodesPerLayer(uint layer);
uint   weightsPerLayerNode(uint layer);
double g(double x);
double netInput(uint layer, uint node);
double deltaTerm(uint layer, uint node, uint p);
void   calculateOutputs();
void   calculateDeltaTerms(uint p);
void   calculateGradient();
void   calculateDeltaWeightChange();
void   calculateTotalError();
int    isOutputEqual(uint p);
int    isConverged(uint epoch);
void   initializeWeights();
void   resetBatchMode();
void   shufflePatternOrder();
double frandUniform();
double frandRange(double a, double b);
void   lowercase(char* str);
uint   getFlag(uint f);
void   setFlag(uint f, uint b);
void   generateNMNEncoder();
void   printHelp();
void   printModel();

//-------------------------------------
// GLOBALS
//-------------------------------------

NNModel  g_model;    // neural net
DataSet  g_dataset;  // training, validation or testing data
Learning g_learning; // learning parameters
uint     g_flags;    // program options

//-------------------------------------
// FUNCTIONS
//-------------------------------------

int main(int argc, char** argv)
{
  if (!initialize(argc,argv)) return 1;

  if (getFlag(CMD_CREATE))
  {
    if (!createModel()) return 1;
    if (!saveModel())   return 1;
    printModel();
  }
  else if (getFlag(CMD_TRAIN))
  {
    if (!loadModel())   return 1;
    printModel();
    if (!loadDataSet()) return 1;
    if (!trainModel())  return 1;
    if (!saveModel())   return 1;
    printModel();
  }
  else if (getFlag(CMD_TEST))
  {
    if (!loadModel())   return 1;
    printModel();
    if (!loadDataSet()) return 1;
    if (!testModel())   return 1;
  }
  else if (getFlag(CMD_VIEW))
  {
    if (!loadModel())   return 1;
    printModel();
  }
  else if (getFlag(CMD_NMN))
  {
    generateNMNEncoder();
  }

  cleanup();
  return 0;
}

int initialize(int argc, char** argv)
{
  uint i,len;

  // program initialization
  memset(&g_model,   0, sizeof(NNModel));
  memset(&g_dataset, 0, sizeof(DataSet));
  memset(&g_learning,0, sizeof(Learning));
  g_flags = 0;

  // program defaults
  g_learning.method  = LEARN_GD;    // Gradient Descent learning by default
  g_learning.cnvgerr = CONVG_ERROR; // default classification error
  g_learning.terror  = 1.0;         // maximum total error
  g_model.error      = 1.0;         // maximum classification
  g_model.best       = 1.0;         // maximum classification

  // quick out
  if (argc<2) {printHelp(); return false;}

  // parse parameters
  for (i=1; i<argc; i++)
  {
    lowercase(argv[i]);

    // help message
    if (strcmp(argv[i],"-help")==0 || strcmp(argv[i],"-h")==0)
    {
      printHelp();
      return false;
    }

    // CMD - create model
    else if (strcmp(argv[i],"-create")==0 || strcmp(argv[i],"-c")==0)
    {
      setFlag(CMD_CREATE, true);

      // model name
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting model filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_model.filename = calloc(len+1, sizeof(char));
      memcpy(g_model.filename, argv[i], len);
      g_model.filename[len] = '\0';

      // number of layers
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of layers (>= 3).\n");
        return false;
      }
      g_model.numLayers = (uchar)strtoul(argv[i], 0, 10);
      g_model.numLayers = max(min(g_model.numLayers,UCHAR_MAX),3);

      // number of input nodes - dimensionality of input
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of input nodes (>= 1).\n");
        return false;
      }
      g_model.numINodes = (uchar)strtoul(argv[i], 0, 10);
      g_model.numINodes = max(min(g_model.numINodes,UCHAR_MAX),1);
      g_model.numINodes++; // account for bias node

      // number of hidden nodes
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of hidden nodes (>= 1).\n");
        return false;
      }
      g_model.numHNodes = (uchar)strtoul(argv[i], 0, 10);
      g_model.numHNodes = max(min(g_model.numHNodes,UCHAR_MAX),1);
      g_model.numHNodes++; // account for bias node

      // number of output nodes
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of output nodes (>= 1).\n");
        return false;
      }
      g_model.numONodes = (uchar)strtoul(argv[i], 0, 10);
      g_model.numONodes = max(min(g_model.numONodes,UCHAR_MAX),1);
    }

    // CMD - train model
    else if (strcmp(argv[i],"-train")==0 || strcmp(argv[i],"-t")==0)
    {
      setFlag(CMD_TRAIN, true);

      // model filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting model filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_model.filename = calloc(len+1, sizeof(char));
      memcpy(g_model.filename, argv[i], len);
      g_model.filename[len] = '\0';

      // training set filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting dataset filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_dataset.filename = calloc(len+1, sizeof(char));
      memcpy(g_dataset.filename, argv[i], len);
      g_dataset.filename[len] = '\0';

      // number of epochs
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting max number of epochs (>= 1).\n");
        return false;
      }
      g_learning.epochs = (uint)strtoul(argv[i], 0, 10);
      g_learning.epochs = max(min(g_learning.epochs,UINT_MAX),1);

      // type of back-propogation learning method
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting back-propogation learning method (0-%u).\n", MAX_LEARN_METHODS-1);
        return false;
      }
      g_learning.method = (uchar)strtoul(argv[i], 0, 10);
      g_learning.method = max(min(g_learning.method,MAX_LEARN_METHODS-1),0);

      // learning rate - eta
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting value for learning rate eta (> 0).\n");
        return false;
      }
      g_learning.eta = (double)strtod(argv[i], 0);
      g_learning.eta = max(min(g_learning.eta,1000),0.01);

      // momentum rate - alpha
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting value for momentum rate alpha (0-1).\n");
        return false;
      }
      g_learning.alpha = (double)strtod(argv[i], 0);
      g_learning.alpha = max(min(g_learning.alpha,1),0);
    }

    // CMD - run/test model
    else if (strcmp(argv[i],"-run")==0 || strcmp(argv[i],"-r")==0)
    {
      setFlag(CMD_TEST, true);

      // model filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting model filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_model.filename = calloc(len+1, sizeof(char));
      memcpy(g_model.filename, argv[i], len);
      g_model.filename[len] = '\0';

      // test set filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting dataset filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_dataset.filename = calloc(len+1, sizeof(char));
      memcpy(g_dataset.filename, argv[i], len);
      g_dataset.filename[len] = '\0';
    }

    // CMD - view model
    else if (strcmp(argv[i],"-view")==0 || strcmp(argv[i],"-v")==0)
    {
      setFlag(CMD_VIEW, true);

      // model filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting model filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_model.filename = calloc(len+1, sizeof(char));
      memcpy(g_model.filename, argv[i], len);
      g_model.filename[len] = '\0';
    }
    
    // CMD - generate encoder dataset
    else if (strcmp(argv[i],"-nmn")==0)
    {
      setFlag(CMD_NMN, true);

      // dataset filename
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting dataset filename.\n");
        return false;
      }
      len = min(strlen(argv[i]), USHRT_MAX-1);
      g_dataset.filename = calloc(len+1, sizeof(char));
      memcpy(g_dataset.filename, argv[i], len);
      g_dataset.filename[len] = '\0';

      // # of bits in encoder (and # of nodes in Input & Output layers (w/out bias))
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of encoder bits (>= 1).\n");
        return false;
      }
      g_dataset.dimInput  = (uchar)strtoul(argv[i], 0, 10);
      g_dataset.dimInput  = max(min(g_dataset.dimInput,UCHAR_MAX),1);
      g_dataset.dimOutput = g_dataset.dimInput;

      // # of patterns to generate
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting number of patterns (>= 1).\n");
        return false;
      }
      g_dataset.numPatterns = (uint)strtoul(argv[i], 0, 10);
      g_dataset.numPatterns = max(min(g_dataset.numPatterns,UINT_MAX),1);
    }

    // PSNG randomizing seed
    else if (strcmp(argv[i],"-seed")==0 || strcmp(argv[i],"-s")==0)
    {
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting value for psuedo random number seed.\n");
        return false;
      }
      g_model.seed = (uint)strtoul(argv[i], 0, 10);
    }

    // additive training
    else if (strcmp(argv[i],"-additive")==0 || strcmp(argv[i],"-add")==0)
    {
      setFlag(ADDITIVE, true);
    }

    // show learning results
    else if (strcmp(argv[i],"-show_learn")==0)
    {
      setFlag(SHOW_LEARN, true);
    }

    // show network model
    else if (strcmp(argv[i],"-show_model")==0)
    {
      setFlag(SHOW_MODEL, true);
    }

    // use classification error as convergence criteria
    else if (strcmp(argv[i],"-error")==0 || strcmp(argv[i],"-e")==0)
    {
      setFlag(USE_ERROR, true);

      // threshold
      if (++i==argc)
      {
        printf("[ERROR][ARGS]: Expecting classification error limit for convergence (0-1).\n");
        return false;
      }
      g_learning.cnvgerr = (double)strtod(argv[i], 0);
      g_learning.cnvgerr = max(min(g_learning.cnvgerr,1),0);
    }

    // XOR model from AoNNiE book (to verify program correctness)
    else if (strcmp(argv[i],"-xor")==0)
    {
      setFlag(XOR_MODEL, true);
    }

    // drop-through
    else
    {
      printf("[ERROR][ARGS]: Unexpected argument '%s'.\n", argv[i]);
      return false;
    }
  }

  // ---------------

  // ensure that model filename was specified
  if (!g_model.filename && !getFlag(CMD_NMN))
  {
    printf("[ERROR][ARGS]: No model filename specified.\n");
    return false;
  }

  // ensure that dataset filename was specified, if necessary
  if (!g_dataset.filename && (getFlag(CMD_TRAIN) || getFlag(CMD_TEST) || getFlag(CMD_NMN)))
  {
    printf("[ERROR][ARGS]: No dataset filename specified.\n");
    return false;
  }

  // initial PRN
  if (!g_model.seed)
  {
#ifdef WIN32
    g_model.seed = (uint)time(0);
#else
    struct timeval tim;
    gettimeofday(&tim, 0);
    g_model.seed = (uint)((tim.tv_sec+(tim.tv_usec/1000000.0))*1000000.0);
#endif
  }
  srand(g_model.seed);

  // use XOR model example in AoNNiE book
  if (getFlag(XOR_MODEL))
  {
    g_model.numLayers = 3;
    g_model.numONodes = 1;
    g_model.numHNodes = 3;
    g_model.numINodes = 3;
    g_learning.method = LEARN_GD;
    g_learning.eta    = 1;
  }

  return true;
}

int createModel()
{
  uint i,j,ni,nm;

  // allocate neurons
  g_model.nodes          = (NNNode**)calloc(g_model.numLayers, sizeof(NNNode*));
  g_model.nodes[I_LAYER] = (NNNode*)calloc(I_COUNT, sizeof(NNNode));
  g_model.nodes[O_LAYER] = (NNNode*)calloc(O_COUNT, sizeof(NNNode));
  for (j=1; j<O_LAYER; j++) g_model.nodes[j] = (NNNode*)calloc(H_COUNT, sizeof(NNNode));

  // allocate weights (recall no weights eminating from output layer)
  for (j=0; j<O_LAYER; j++)
    for (i=0; i<nodesPerLayer(j); i++)
      g_model.nodes[j][i].weights = (NNWeight*)calloc(weightsPerLayerNode(j), sizeof(NNWeight));

  // initialization of neurons (weights initialized at start of training)
  for (j=0; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);
    for (i=0; i<ni; i++)
      g_model.nodes[j][i].output = (j!=O_LAYER && i==ni-1)?1:0;
  }

  return true;
}

int trainModel()
{
  uint   i,p,e;
  uchar  converged;
  time_t now,then=0;
  
  // -----------------------------------
  // Back-Propagation Learning Algorithm
  // -----------------------------------

  // Step 1: initialize weights & p = first pattern
  p=0; // pattern counter
  e=0; // epoch counter
  if (!getFlag(ADDITIVE))   initializeWeights();
  if (!getFlag(XOR_MODEL))  shufflePatternOrder();
  if (!getFlag(SHOW_LEARN)) {printf("Train: "); fflush(stdout);}  

  // for each pattern
  for (; p<g_dataset.numPatterns; p++)
  {
    // Step 2: present training pattern input to input layer
    for (i=0; i<I_COUNT-1; i++)
      g_model.nodes[I_LAYER][i].output = PATTERN(p).inputs[i];

    // Step 3: propagate the pattern forward through all neurons
    calculateOutputs();

    // Step 4: check output, if output = expected, goto Step 7
    if (!isOutputEqual(p))
    {
      // Step 5: propagate backwards, calculate delta terms for all neurons
      calculateDeltaTerms(p);

      // Step 6: adjust weights according to delta learning rules (STOCHASTIC)
      calculateGradient();
      if (LEARNING(LEARN_GD)) calculateDeltaWeightChange();
    }

    // Step 7: if p = PT, check convergence criteria, goto Step 2, next epoch
    //         else       goto Step 2
    if (p == g_dataset.numPatterns-1)
    {
      // Step 6: adjust weights according to delta learning rules (BATCH)
      if (LEARNING(LEARN_GDM) || LEARNING(LEARN_RPROP)) calculateDeltaWeightChange();

      // calculate average sum of squared errors
      calculateTotalError();

      // calculate convergence
      converged = isConverged(e);

      // show learning (otherwise display progress bar...)
      if (getFlag(SHOW_LEARN)) printf("%05u: class(%u) error(%5.3f) best(%5.3f)\n", e,g_learning.correct,g_model.error,g_model.best);
      else
      {
        now = time(0);
        if (now-then>=1)
        {
          printf("."); fflush(stdout);
          then = now;
        }
      }

      // break out upon convergence (otherwise setup for next epoch)
      if (converged) break;
      else
      {
        p = 0;
        e++;
        if (LEARNING(LEARN_GDM) || LEARNING(LEARN_RPROP)) resetBatchMode();
        if (!getFlag(XOR_MODEL)) shufflePatternOrder();
      }
    }
  }

  // wrap up training printouts
  if (!getFlag(SHOW_LEARN)) printf("\n");
  printf("Convg: (%s) class(%u) error(%5.3f) best(%5.3f) epochs(%u)\n",
    (e>=g_learning.epochs-1)?"No":"Yes", g_learning.correct, g_model.error, g_model.best, (e+1));

  return true;
}

int testModel()
{
  FILE*  file;
  uint   i,p,truenode,highnode,correct;
  uchar  converged;
  char   ofilename[64];
  double high;

  // open output file
  sprintf(ofilename, "%s_output.csv", g_model.filename);
  if (!(file = fopen(ofilename, "w+")))
  {
    printf("[ERROR]: Could not create/open output file: '%s'\n", ofilename);
    return false;
  }
  
  // init
  correct = 0;

  // output header
  for (i=0; i<I_COUNT-1; i++)
    fprintf(file, "I-%u,", i);
  fprintf(file, "Desired,");
  for (i=0; i<O_COUNT; i++)
    fprintf(file, "O-%u,", i);
  fprintf(file, "Actual,");
  fprintf(file, "Correct\n");

  // for each test pattern
  for (p=0; p<g_dataset.numPatterns; p++)
  {
    // present pattern input to input layer (and output to file)
    for (i=0; i<I_COUNT-1; i++)
    {
      g_model.nodes[I_LAYER][i].output = PATTERN(p).inputs[i];
      fprintf(file, "%g,", PATTERN(p).inputs[i]);
    }
    fprintf(file, "%u,", PATTERN(p).output);

    // propagate the pattern forward through all neurons
    calculateOutputs();

    // for yes-no neural networks w/ one output
    if (O_COUNT == 1)
    {
      converged = (((PATTERN(p).output==1) && (g_model.nodes[O_LAYER][0].output>CONVG_1_LIM)) ||
                   ((PATTERN(p).output==0) && (g_model.nodes[O_LAYER][0].output<CONVG_0_LIM)));
      fprintf(file, "%g,", g_model.nodes[O_LAYER][0].output);
      if      (g_model.nodes[O_LAYER][0].output>CONVG_1_LIM) fprintf(file, "%d,", 1);
      else if (g_model.nodes[O_LAYER][0].output<CONVG_0_LIM) fprintf(file, "%d,", 0);
      else    fprintf(file, "%d,", -1);
      fprintf(file, "%d\n", converged);
    }
  
    // for NNs w/ multiple outputs
    else
    {
      converged = true;

      // the expected output corresponds to the output node that converges high
      truenode = PATTERN(p).output;

      // ensure truth node is "true" (1.0 or high convergence value)
      if (g_model.nodes[O_LAYER][truenode].output <= CONVG_1_LIM) converged = false;

      // ensure the remaining output nodes are "false" (0.0 or low convergence value)
      highnode = -1;
      high     = 0;
      for (i=0; i<O_COUNT; i++)
      {
        if (g_model.nodes[O_LAYER][i].output > high)
        {
          high     = g_model.nodes[O_LAYER][i].output;
          highnode = i;
        }
        fprintf(file, "%g,", g_model.nodes[O_LAYER][i].output);
        if (i == truenode) continue;
        if (g_model.nodes[O_LAYER][i].output >= CONVG_0_LIM) converged = false;
      }
      if      (converged)        fprintf(file, "%d,", truenode);
      else if (high>CONVG_1_LIM) fprintf(file, "%d,", highnode);
      else                       fprintf(file, "-1,");
      fprintf(file, "%d\n", converged);
    }
    
    // increment correct classification count
    if (converged) correct++;
  }
  
  // print correct classifications
  printf("Right: %u\n", correct);

  fclose(file);
  return true;
}

int saveModel()
{
  FILE* file;
  uint  i,j,m,ni,nm;

  if (!(file = fopen(g_model.filename, "w+")))
  {
    printf("[ERROR]: Could not create/open BPNN model file: '%s'\n", g_model.filename);
    return false;
  }

  // write template information
  fwrite(&g_model.seed,     sizeof(uint),  1,file);
  fwrite(&g_model.error,    sizeof(double),1,file);
  fwrite(&g_model.best,     sizeof(double),1,file);
  fwrite(&g_model.numLayers,sizeof(uchar), 1,file);
  fwrite(&g_model.numINodes,sizeof(uchar), 1,file);
  fwrite(&g_model.numHNodes,sizeof(uchar), 1,file);
  fwrite(&g_model.numONodes,sizeof(uchar), 1,file);

  // write neurons and weights
  for (j=0; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);
    for (i=0; i<ni; i++)
    {
      //fwrite(&g_model.nodes[j][i].output,sizeof(double),1,file);
      for (m=0; m<nm; m++)
        fwrite(&g_model.nodes[j][i].weights[m].w,sizeof(double),1,file);
    }
  }

  fclose(file);
  return true;
}

int loadModel()
{
  FILE*   file;
  uint    i,j,m,ni,nm;

  if (!(file = fopen(g_model.filename, "r")))
  {
    printf("[ERROR]: Could not find BPNN model file: '%s'\n", g_model.filename);
    return false;
  }

  // read template information
  fread(&g_model.seed,     sizeof(uint),  1,file);
  fread(&g_model.error,    sizeof(double),1,file);
  fread(&g_model.best,     sizeof(double),1,file);
  fread(&g_model.numLayers,sizeof(uchar), 1,file);
  fread(&g_model.numINodes,sizeof(uchar), 1,file);
  fread(&g_model.numHNodes,sizeof(uchar), 1,file);
  fread(&g_model.numONodes,sizeof(uchar), 1,file);

  // allocate neurons
  g_model.nodes          = (NNNode**)calloc(g_model.numLayers, sizeof(NNNode*));
  g_model.nodes[I_LAYER] = (NNNode*)calloc(I_COUNT, sizeof(NNNode));
  g_model.nodes[O_LAYER] = (NNNode*)calloc(O_COUNT, sizeof(NNNode));
  for (j=1; j<O_LAYER; j++) g_model.nodes[j] = (NNNode*)calloc(H_COUNT, sizeof(NNNode));

  // allocate weights (recall no weights eminating from output layer)
  for (j=0; j<g_model.numLayers-1; j++)
    for (i=0; i<nodesPerLayer(j); i++)
      g_model.nodes[j][i].weights = (NNWeight*)calloc(weightsPerLayerNode(j), sizeof(NNWeight));

  // read neurons and weights
  for (j=0; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);
    for (i=0; i<ni; i++)
    {
      //fread(&g_model.nodes[j][i].output,sizeof(double),1,file);
      for (m=0; m<nm; m++)
        fread(&g_model.nodes[j][i].weights[m].w,sizeof(double),1,file);
    }
  }

  // initialize bias node outpus (weights initialized at start of training)
  for (j=0; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);
    for (i=0; i<ni; i++)
      g_model.nodes[j][i].output = (j!=O_LAYER && i==ni-1)?1:0;
  }

  // re-seed PRNG w/ model seed
  srand(g_model.seed);

  fclose(file);
  return true;
}

int loadDataSet()
{
  FILE*  file;
  char   buffer[512];
  char*  tok;
  uint   i,p;

  if (!(file = fopen(g_dataset.filename, "r")))
  {
    printf("[ERROR]: Could not find dataset file: '%s'\n", g_dataset.filename);
    return false;
  }

  // read dimensionality of input
  if (!fgets(buffer, 512, file))
  {
    printf("[ERROR]: Failed reading dataset dimensionality of input.\n");
    fclose(file); return false;
  }
  g_dataset.dimInput = (uchar)strtoul(buffer, 0, 10);
  if (g_dataset.dimInput < 1)
  {
    printf("[ERROR]: Dataset dimensionality of input '%u' is invalid.\n", g_dataset.dimInput);
    fclose(file); return false;
  }

  // read dimensionality of output
  if (!fgets(buffer, 512, file))
  {
    printf("[ERROR]: Failed reading dataset dimensionality of output.\n");
    fclose(file); return false;
  }
  g_dataset.dimOutput = (uchar)strtoul(buffer, 0, 10);
  if (g_dataset.dimOutput < 1)
  {
    printf("[ERROR]: Dataset dimensionality of output '%u' is invalid.\n", g_dataset.dimOutput);
    fclose(file); return false;
  }

  // MAKE SURE TRAINING/TEST DATA IS SAME DIMENSIONALITY AS MODEL
  if (g_dataset.dimInput != I_COUNT-1)
  {
    printf("[ERROR]: Dataset input does not have same dimensionality as model input.\n");
    fclose(file); return false;
  }
  if (g_dataset.dimOutput != O_COUNT)
  {
    printf("[ERROR]: Dataset output does not have same dimensionality as model output.\n");
    fclose(file); return false;
  }
  // MAKE SURE TRAINING/TEST DATA IS SAME DIMENSIONALITY AS MODEL

  // read number of patterns
  if (!fgets(buffer, 512, file))
  {
    printf("[ERROR]: Failed reading dataset number of patterns.\n");
    fclose(file); return false;
  }
  g_dataset.numPatterns = (uint)strtoul(buffer, 0, 10);
  if (g_dataset.numPatterns < 1)
  {
    printf("[ERROR]: Dataset number of patterns '%u' is invalid.\n", g_dataset.numPatterns);
    fclose(file); return false;
  }

  // allocate patterns
  g_dataset.patterns = (Pattern*)calloc(g_dataset.numPatterns, sizeof(Pattern));
  g_dataset.order    = (uint*)calloc(g_dataset.numPatterns, sizeof(uint));

  // read each pattern
  for (p=0; p<g_dataset.numPatterns; p++)
  {
    if (!fgets(buffer, 512, file))
    {
      printf("[ERROR]: Failed reading dataset pattern '%u'.\n", p);
      fclose(file); return false;
    }

    // allocate pattern input
    g_dataset.patterns[p].inputs = (double*)calloc(g_dataset.dimInput, sizeof(double));

    // read inputs
    for (i=0; i<g_dataset.dimInput; i++)
    {
      if (!(tok = (char*)strtok((i!=0)?0:buffer, ",")))
      {
        printf("[ERROR]: Failed reading dataset pattern '%u' input '%u'.\n", p, i);
        fclose(file); return false;
      }
      g_dataset.patterns[p].inputs[i] = (double)strtod(tok, 0);
    }

    // read expected output
    if (!(tok = (char*)strtok(0, DELIMITERS)))
    {
      printf("[ERROR]: Failed reading dataset pattern '%u' output.\n", p);
      fclose(file); return false;
    }
    g_dataset.patterns[p].output = (uint)strtol(tok, 0, 10);
    if (g_dataset.patterns[p].output > g_dataset.dimOutput)
    {
      printf("[ERROR]: Dataset pattern '%u' output '%u' is beyond defined dimensionality '%u'.\n",
        p, g_dataset.patterns[p].output, g_dataset.dimOutput);
      fclose(file); return false;
    }
  }

  // initial pattern access order (before any reordering)
  for (p=0; p<g_dataset.numPatterns; p++)
    g_dataset.order[p] = p;

  fclose(file);
  return true;
}

uint nodesPerLayer(uint layer)
{
  if      (layer >  O_LAYER) return 0;
  else if (layer == O_LAYER) return O_COUNT;
  else if (layer == I_LAYER) return I_COUNT;
  else                       return H_COUNT;
}

uint weightsPerLayerNode(uint layer)
{
  if      (layer >= O_LAYER)   return 0;         // no weights from output layer
  else if (layer == O_LAYER-1) return O_COUNT;   // output layer has no bias node
  else                         return H_COUNT-1; // no weights going to bias nodes
}

double g(double x)
{
  return pow((1 + exp(-x)), -1.0);
}

double netInput(uint layer, uint node)
{
  uint   k,nk;
  double sum;

  sum    = 0;
  layer -= 1; // gather data from previous layer into 'node'
  nk     = nodesPerLayer(layer);

  // for all nodes in the previous layer
  for (k=0; k<nk; k++)
    sum += (g_model.nodes[layer][k].weights[node].w * g_model.nodes[layer][k].output);

  return sum;
}

double deltaTerm(uint layer, uint node, uint p)
{
  uint   m,nm;
  double delta;

  delta = 0;

  // output layer delta term is error between expected and actual
  if (layer == O_LAYER)
  {
    if (O_COUNT == 1) delta = (PATTERN(p).output - g_model.nodes[layer][node].output);
    else
    {
      if (PATTERN(p).output == node) delta = (1.0 - g_model.nodes[layer][node].output);
      else                           delta = (0.0 - g_model.nodes[layer][node].output);
    }
  }

  // hidden layer delta terms are sum of weights times higher layer delta term
  else
  {
    nm = nodesPerLayer(layer+1);

    // bias nodes not involved in these calculations (output layer has no bias)
    if (layer+1 < O_LAYER) nm--;

    // for all weights in current layer to nodes in higher layer
    for (m=0; m<nm; m++)
      delta += (g_model.nodes[layer][node].weights[m].w * g_model.nodes[layer+1][m].delta);
  }

  return ((g_model.nodes[layer][node].output)*(1-g_model.nodes[layer][node].output)*delta);
}

void calculateOutputs()
{
  uint i,j,ni;

  // for all layers execept input layer
  for (j=1; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);

    // bias nodes not involved in these calculations (output layer has no bias)
    if (j < O_LAYER) ni--;
  
    // for all nodes in the current layer (except bias)
    for (i=0; i<ni; i++)
      g_model.nodes[j][i].output = g(netInput(j,i));
  }
}

void calculateDeltaTerms(uint p)
{
  uint i,ni;
  int  j;

  // for all layers execept input layer
  for (j=O_LAYER; j>I_LAYER; j--)
  {
    ni = nodesPerLayer(j);

    // other than output layer (which has no bias node),
    //  bias nodes do not calculate delta terms
    if (j < O_LAYER) ni--;

    // for all nodes in the current layer (except bias)
    for (i=0; i<ni; i++)
      g_model.nodes[j][i].delta = deltaTerm(j,i,p);
  }
}

void calculateGradient()
{
  uint   i,m,ni,nm;
  int    j;
  double grad;

  // for all layers execept output layer
  for (j=O_LAYER-1; j>=0; j--)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);

    // for all nodes per layer
    for (i=0; i<ni; i++)
    {
      // for all weights per node
      for (m=0; m<nm; m++)
      {
        grad = -1.0 * g_model.nodes[j+1][m].delta * g_model.nodes[j][i].output;
        if      (LEARNING(LEARN_GD))    g_model.nodes[j][i].weights[m].grad  = grad;
        else if (LEARNING(LEARN_GDM))   g_model.nodes[j][i].weights[m].grad += grad;
        else if (LEARNING(LEARN_RPROP)) g_model.nodes[j][i].weights[m].grad += grad;
      }
    }
  }
}

void calculateDeltaWeightChange()
{
  uint      i,m,ni,nm;
  int       j;
  double    gnew;
  NNWeight* pweight;

  // for all layers execept output layer
  for (j=O_LAYER-1; j>=0; j--)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);

    // for all nodes per layer
    for (i=0; i<ni; i++)
    {
      // for all weights per node
      for (m=0; m<nm; m++)
      {
        pweight = &(g_model.nodes[j][i].weights[m]);

        // Gradient Descent
        if (LEARNING(LEARN_GD))
        {
          pweight->dw = -1.0 * g_learning.eta * pweight->grad;
          pweight->w += pweight->dw;
        }

        // Gradient Descent w/ Momentum
        else if (LEARNING(LEARN_GDM))
        {
          pweight->dw = (-1.0 * g_learning.eta * pweight->grad) + (g_learning.alpha * pweight->dwLast);
          pweight->w += pweight->dw;
        }

        // Resillient Propagation (RPROP)
        else if (LEARNING(LEARN_RPROP))
        {
          gnew = (pweight->gradLast * pweight->grad);

          if (gnew > 0)
          {
            pweight->sz = min((pweight->szLast * 1.2), 50.0);
            pweight->dw = (-sign(pweight->grad)) * pweight->sz;
            pweight->w += pweight->dw;
          }
          else if (gnew < 0)
          {
            pweight->sz = max((pweight->szLast * 0.5), 1.0e-6);
            pweight->w -= pweight->dwLast;
            pweight->grad = 0;
          }
          else //if ((pweight->gradLast * pweight->grad) == 0)
          {
            pweight->dw = (-sign(pweight->grad)) * pweight->sz;
            pweight->w += pweight->dw;
          }

          pweight->gradLast = pweight->grad;
          pweight->szLast   = pweight->sz;
          pweight->dwLast   = pweight->dw;
        }
      }
    }
  }
}

void calculateTotalError()
{
  uint i,p;
  double diff, sum;

  // reset error, we are going to recalculate it
  g_learning.terror = 0;

  // for all patterns
  for (p=0; p<g_dataset.numPatterns; p++)
  {
    // present training pattern input to input layer
    for (i=0; i<I_COUNT-1; i++)
      g_model.nodes[I_LAYER][i].output = PATTERN(p).inputs[i];

    // propagate the pattern forward through all neurons
    calculateOutputs();

    sum = 0;

    // calculate sum of squared difference between expected and actual output
    if (O_COUNT == 1)
    {
      diff  = (PATTERN(p).output - g_model.nodes[O_LAYER][0].output);
      diff *= diff;       // squared error
      sum  += diff;       // sum of squared error
    }
    else
    {
      for (i=0; i<O_COUNT; i++)
      {
        if (PATTERN(p).output == i) diff = (1.0 - g_model.nodes[O_LAYER][i].output);
        else                        diff = (0.0 - g_model.nodes[O_LAYER][i].output);
        diff *= diff;     // squared error
        sum  += diff;     // sum of squared error
      }
      sum /= (double)O_COUNT;

      // this constant doesn't seem to be necessary for some reason
      //g_learning.terror *= 0.5; // 1/2 constant in global error function
    }

    // sum of squared error across all patterns
    g_learning.terror += sum;
  }

  // average sum of squared errors across all patterns
  g_learning.terror /= (double)g_dataset.numPatterns;
}

int isOutputEqual(uint p)
{
  // for yes-no neural networks w/ one output
  if (O_COUNT == 1)
  {
    return (fabs(PATTERN(p).output - g_model.nodes[O_LAYER][0].output) <= DBL_EPSILON);
  }

  // for NNs w/ multiple outputs
  else
  {
    uint i, truenode;

    // the expected output corresponds to the output node that converges high
    truenode = PATTERN(p).output;

    // ensure truth node is "true" (1.0 or high convergence value)
    if (fabs(g_model.nodes[O_LAYER][truenode].output - 1.0) > DBL_EPSILON) return false;

    // ensure the remaining output nodes are "false" (0.0 or low convergence value)
    for (i=0; i<O_COUNT; i++)
    {
      if (i == truenode) continue;
      if (fabs(g_model.nodes[O_LAYER][i].output - 0.0) > DBL_EPSILON) return false;
    }

    return true;
  }
}

int isConverged(uint epoch)
{
  uint i,p,truenode,converged;

  // XOR model only runs one epoch (just for testing)
  if (getFlag(XOR_MODEL)) return true;

  // reset
  g_learning.correct = 0;

  // check classification of all patterns
  for (p=0; p<g_dataset.numPatterns; p++)
  {
    converged = true;

    // present training pattern input to input layer
    for (i=0; i<I_COUNT-1; i++)
      g_model.nodes[I_LAYER][i].output = PATTERN(p).inputs[i];

    // propagate the pattern forward through all neurons
    calculateOutputs();

    // for yes-no neural networks w/ one output
    if (O_COUNT == 1)
    {
      converged = (((PATTERN(p).output==1) && (g_model.nodes[O_LAYER][0].output>CONVG_1_LIM)) ||
                   ((PATTERN(p).output==0) && (g_model.nodes[O_LAYER][0].output<CONVG_0_LIM)));
    }

    // for NNs w/ multiple outputs
    else
    {
      // the expected output corresponds to the output node that converges high
      truenode = PATTERN(p).output;
  
      // truth node should be "true" (1.0 or high threshold)
      if (!(g_model.nodes[O_LAYER][truenode].output > CONVG_1_LIM))
        converged = false;

      // remaining nodes should be "false" (0.0 or low threshold)
      for (i=0; i<O_COUNT; i++)
      {
        if (i == truenode) continue;
        if (!(g_model.nodes[O_LAYER][i].output < CONVG_0_LIM))
          converged = false;
      }
    }

    // one for the money!
    if (converged) g_learning.correct++;
  }

  // calculate classification error
  g_model.error = 1.0-((double)g_learning.correct/(double)g_dataset.numPatterns);
  if (g_model.error < g_model.best) g_model.best = g_model.error;

  // converge using classification error
  if (getFlag(USE_ERROR) && (g_model.error < g_learning.cnvgerr)) return true;

  // converge if all patterns correctly classified or max epochs reached
  return ((g_learning.correct == g_dataset.numPatterns) || (epoch >= g_learning.epochs-1));
}

void initializeWeights()
{
  uint i,j,m,ni,nm;

  // for XOR example model, hardcode the weights to match AoNNiE book
  if (getFlag(XOR_MODEL))
  {
    // output layer - no weights

    // hidden layer
    g_model.nodes[1][0].weights[0].w = 0.2;
    g_model.nodes[1][1].weights[0].w = 0.3;
    g_model.nodes[1][2].weights[0].w = 0.1; // from bias node
  
    // input layer
    g_model.nodes[I_LAYER][0].weights[0].w =  0.2;
    g_model.nodes[I_LAYER][0].weights[1].w =  0.2;
    g_model.nodes[I_LAYER][1].weights[0].w = -0.4;
    g_model.nodes[I_LAYER][1].weights[1].w = -0.1;
    g_model.nodes[I_LAYER][2].weights[0].w =  0.1; // from bias node
    g_model.nodes[I_LAYER][2].weights[1].w =  0.3; // from bias node
  }

  // select an appropriate starting weight value
  else
  {
    for (j=0; j<g_model.numLayers; j++)
    {
      ni = nodesPerLayer(j);
      nm = weightsPerLayerNode(j);
      for (i=0; i<ni; i++)
      {
        for (m=0; m<nm; m++)
        {
          g_model.nodes[j][i].weights[m].w        = frandRange(-0.5,0.5);
          g_model.nodes[j][i].weights[m].grad     = 0;
          g_model.nodes[j][i].weights[m].gradLast = 0;
          g_model.nodes[j][i].weights[m].sz       = 0.1;
          g_model.nodes[j][i].weights[m].szLast   = 0.1;
          g_model.nodes[j][i].weights[m].dw       = 0;
          g_model.nodes[j][i].weights[m].dwLast   = 0;
        }
      }
    }
  }
}

void resetBatchMode()
{
  uint i,j,m,ni,nm;

  for (j=0; j<g_model.numLayers; j++)
  {
    ni = nodesPerLayer(j);
    nm = weightsPerLayerNode(j);
    for (i=0; i<ni; i++)
      for (m=0; m<nm; m++)
      {
        g_model.nodes[j][i].weights[m].grad     = 0;
        g_model.nodes[j][i].weights[m].sz       = 0.1;
        g_model.nodes[j][i].weights[m].dw       = 0;
      }
  }
}

void shufflePatternOrder()
{
  uint i,p1,p2,tmp;
  for (i=0; i<5; i++)
  {
    for (p1=0; p1<g_dataset.numPatterns; p1++)
    {
      p2 = rand()%g_dataset.numPatterns;
      tmp                 = g_dataset.order[p1];
      g_dataset.order[p1] = p2;
      g_dataset.order[p2] = tmp;
    }
  }
}

double frandUniform()
{
  return rand()/(double)RAND_MAX;
}

double frandRange(double a, double b)
{
  return ((b-a)*frandUniform())+a;
}

uint getFlag(uint f)
{
  return (g_flags & f);
}

void setFlag(uint f, uint b)
{
  if (b) g_flags |=  f;
  else   g_flags &= ~f;
}

void lowercase(char* str)
{
  unsigned int i,len = strlen(str);
  for (i=0; i<len; i++) if (str[i] >= 65 && str[i] <= 90) str[i] += 32;
}

void generateNMNEncoder()
{
  FILE*  file;
  uint   i,j,r;

  // open file
  if (!(file = fopen(g_dataset.filename, "w+")))
  {
    printf("[ERROR]: Could not create/open dataset file: '%s'\n", g_dataset.filename);
    return;
  }

  // header
  fprintf(file, "%u\n", g_dataset.dimInput);
  fprintf(file, "%u\n", g_dataset.dimInput);
  fprintf(file, "%u\n", g_dataset.numPatterns);

  // patterns
  for (j=0; j<g_dataset.numPatterns; j++)
  {
    r = rand()%g_dataset.dimInput;
    for (i=0; i<g_dataset.dimInput; i++)
      fprintf(file, "%u,", (i==r)?1:0);
    fprintf(file, "%u\n", r);
  }

  fclose(file);
}

void cleanup()
{
  uint i,j;

  if (g_model.filename)   free(g_model.filename);
  if (g_dataset.filename) free(g_dataset.filename);

  if (g_model.nodes)
  {
    for (j=0; j<g_model.numLayers; j++)
    {
      if (j!=O_LAYER)
      {
        for (i=0; i<nodesPerLayer(j); i++)
          if (g_model.nodes[j][i].weights) free(g_model.nodes[j][i].weights);
      }
      free(g_model.nodes[j]);
    }
    free(g_model.nodes);
  }

  if (g_dataset.order) free(g_dataset.order);
  if (g_dataset.patterns)
  {
    for (j=0; j<g_dataset.numPatterns; j++)
      if (g_dataset.patterns[j].inputs) free(g_dataset.patterns[j].inputs);
    free(g_dataset.patterns);
  }
}

void printHelp()
{
  printf("-----------------------------------------------------\n");
  printf("How to use this program.\n");
  printf("\n");
  printf("One of the following arguments is required:\n");
  printf("  -c|create  (creates MLP BPNN model and saves it)\n");
  printf("  -t|train   (trains existing model and saves it)\n");
  printf("  -r|run     (runs existing model against data set)\n");
  printf("  -v|view    (prints an existing model's information)\n");
  printf("  -n|nmn     (generates N-M-N encoder dataset)\n");
  printf("\n");
  printf("The following arguments are optional:\n");
  printf("  -h|help          (prints this help message)\n");
  printf("  -s|seed  <seed>  (specify PRN seed to repeat results)\n");
  printf("  -e|error <0-1>   (use classification error threshold as convergence)\n");
  printf("  -show_learn      (show learing results in real time)\n");
  printf("  -show_model      (show network model)\n");
  printf("  -add|additive    (additional training to existing weights)\n");
  printf("  -xor             (creates XOR model - trains only 1 epoch)\n");
  printf("\n");
  printf("(C)reate command syntax:\n");
  printf("  -c <name> <layers> <I> <H> <O>\n");
  printf("     <name>     = filename to use for model file\n");
  printf("     <layers>   = # of total layers including Input & Output\n");
  printf("     <I>        = # of nodes in Input Layer\n");
  printf("     <H>        = # of nodes in Hidden Layers\n");
  printf("     <O>        = # of nodes in Output Layer\n");
  printf("\n");
  printf("(T)rain command syntax:\n");
  printf("  -t <name> <dataset> <epochs> <learning> <eta> <alpha>\n");
  printf("     <name>     = filename of model\n");
  printf("     <dataset>  = filename of training data\n");
  printf("     <epochs>   = max # of epochs to run training\n");
  printf("     <learning> = 0|1|2 for GD,GDM,RPROP respectively\n");
  printf("     <eta>      = learning rate (0-?)\n");
  printf("     <alpha>    = momentum rate (0-1) (GDM only)\n");
  printf("\n");
  printf("(R)un command syntax:\n");
  printf("  -r <name> <dataset>\n");
  printf("     <name>     = filename of model file\n");
  printf("     <dataset>  = filename of test data\n");
  printf("\n");
  printf("(V)iew command syntax:\n");
  printf("  -v <name>\n");
  printf("     <name>     = filename of model file\n");
  printf("\n");
  printf("(N)MN command syntax:\n");
  printf("  -n <name> <N> <patterns>\n");
  printf("     <name>     = filename of dataset\n");
  printf("     <N>        = number of nodes in Input & Output layer (w/out bias)\n");
  printf("     <patterns> = number of test patterns to generate\n");
  printf("\n");
  printf("Note that you do not have to account for bias nodes,\n");
  printf("  they will be added-in automatically.  So passing 2\n");
  printf("  for the # of input nodes will ultimately result in\n");
  printf("  3 nodes with the bias node.\n");
  printf("Note that momentum rate (alpha) is only used for GDM\n");
  printf("  training method.\n");
  printf("Note the PRN seed effects anything randomized, such\n");
  printf("  as the initial weights, selection of patterns, etc.,\n");
  printf("  so the exact same model and training can be duplicated\n");
  printf("  by using the same PRN.\n");
  printf("\n");
  printf("Example Usage:\n");
  printf("  > program -c m1 4 2 8 5\n");
  printf("  > program -c m2 4 2 3 5 -s 91768385\n");
  printf("  > program -t m2 CinS500_train.csv 5000 0 1.0 0\n");
  printf("  > program -t m2 CinS500_train2.csv 2000 0 0.85 0 -add\n");
  printf("  > program -r m2 CinS500_test.csv\n");
  printf("  > program -v m1\n");
  printf("  > program -h\n");
  printf("-----------------------------------------------------\n");
}

void printModel()
{
  uint i,m,ni,nm;
  int j;
  char buffer[32];

  printf("-----------------------------------------------------\n");
  printf("Model: '%s'\n",           g_model.filename);
  printf("Seed : %u\n",             g_model.seed);
  printf("Layer: %u\n",             g_model.numLayers);
  printf("Nodes: I=%u H=%u O=%u\n", g_model.numINodes, g_model.numHNodes, g_model.numONodes);
  printf("Error: %.3f (%.3f)\n",    g_model.error, g_model.best);
  printf("-----------------------------------------------------\n");
  if (getFlag(CMD_TEST))
  {
    printf("Data : '%s'\n",         g_dataset.filename);
    printf("-----------------------------------------------------\n");
  }
  if (getFlag(CMD_TRAIN))
  {
    printf("Data : '%s'\n",         g_dataset.filename);
    printf("Epoch: %u\n",           g_learning.epochs);
    printf("Learn: %u\n",           g_learning.method);
    printf("LRate: %.3f\n",         g_learning.eta);
    printf("MRate: %.3f\n",         g_learning.alpha);
    printf("-----------------------------------------------------\n");
  }
  if (getFlag(SHOW_MODEL))
  {
    for (j=O_LAYER; j>=0; j--)
    {
      ni = nodesPerLayer(j);
      nm = weightsPerLayerNode(j);
      for (m=0; m<nm; m++)
      {
        printf("       ");
        for (i=0; i<ni; i++)
        {
          sprintf(buffer, "%.4f", g_model.nodes[j][i].weights[m].w);
          printf("%8s ", buffer);
        }
        printf("\n");
      }
      printf("Lay-%d: ", j);
      for (i=0; i<ni; i++)
      {
        sprintf(buffer, "(%.3f)", g_model.nodes[j][i].output);
        printf("%8s ", buffer);
      }
      printf("\n-----------------------------------------------------\n");
    }
  }
}
