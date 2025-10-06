/*working variables*/
unsigned long lastTime;
double Input, Output, Setpoint;
double outputSum, lastInput;
double kp, ki, kd;
int SampleTime = 1000; //1 sec
double outMin, outMax;
bool inAuto = false;
 
#define MANUAL 0
#define AUTOMATIC 1
 
#define DIRECT 0
#define REVERSE 1
int controllerDirection = DIRECT;
 
#define P_ON_M 0
#define P_ON_E 1
bool pOnE = true, pOnM = false;
double pOnEKp, pOnMKp;


void Compute()
{
   if(!inAuto) return;
   unsigned long now = millis();
   int timeChange = (now - lastTime);
   if(timeChange>=SampleTime)
   {
   
      /*Compute all the working error variables*/	 	
      double error = Setpoint - Input;	 
      double dInput = (Input - lastInput);
      outputSum+= (ki * error);	 
	  
	  /*Add Proportional on Measurement, if P_ON_M is specified*/
	  if(pOnM) outputSum-= pOnMKp * dInput
	  
      if(outputSum > outMax) outputSum= outMax;	 	
      else if(outputSum < outMin) outputSum= outMin;  
   	
   	  /*Add Proportional on Error, if P_ON_E is specified*/ 
	  if(pOnE) Output = pOnEKp * error; 
	  else Output = 0;
   	  
   	  /*Compute Rest of PID Output*/ 
   	  Output += outputSum - kd * dInput; 
   
      if(Output > outMax) Output = outMax;
      else if(Output < outMin) Output = outMin;
 
      /*Remember some variables for next time*/
      lastInput = Input;
      lastTime = now;
   }
}
 
void SetTunings(double Kp, double Ki, double Kd, double pOn)
{
   if (Kp<0 || Ki<0|| Kd<0 || pOn<0 || pOn>1) return;
 
   pOnE = pOn>0; //some p on error is desired;
   pOnM = pOn<1; //some p on measurement is desired;  
  
   double SampleTimeInSec = ((double)SampleTime)/1000;
   kp = Kp;
   ki = Ki * SampleTimeInSec;
   kd = Kd / SampleTimeInSec;
 
  if(controllerDirection ==REVERSE)
   {
      kp = (0 - kp);
      ki = (0 - ki);
      kd = (0 - kd);
   }
   
   pOnEKp = pOn * kp; 
   pOnMKp = (1 - pOn) * kp;
}
 
void SetSampleTime(int NewSampleTime)
{
   if (NewSampleTime > 0)
   {
      double ratio  = (double)NewSampleTime
                      / (double)SampleTime;
      ki *= ratio;
      kd /= ratio;
      SampleTime = (unsigned long)NewSampleTime;
   }
}
 
void SetOutputLimits(double Min, double Max)
{
   if(Min > Max) return;
   outMin = Min;
   outMax = Max;
 
   if(Output > outMax) Output = outMax;
   else if(Output < outMin) Output = outMin;
 
   if(outputSum > outMax) outputSum= outMax;
   else if(outputSum < outMin) outputSum= outMin;
}
 
void SetMode(int Mode)
{
    bool newAuto = (Mode == AUTOMATIC);
    if(newAuto == !inAuto)
    {  /*we just went from manual to auto*/
        Initialize();
    }
    inAuto = newAuto;
}
 
void Initialize()
{
   lastInput = Input;
   outputSum = Output;
   if(outputSum > outMax) outputSum= outMax;
   else if(outputSum < outMin) outputSum= outMin;
}
 
void SetControllerDirection(int Direction)
{
   controllerDirection = Direction;
}
