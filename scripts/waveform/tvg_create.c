#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

//argument 1 is the function to be written into tvg (sin,cos,rand,zeros,ones,step,pulse) 
//argument 2 is frequency
//argument 3 is period (if there is one, otherwise omit)

int rnd(double x);

int main(int argc,char **argv)
{
    if(argv[1] == NULL)
    {
	printf("Usage:tvg_create <sin/cos/rand/zeros/ones/const/step/pulse> <freq> [period]\n");
    }
    else
    {
	double period=2.*M_PI;
	double skip=1;
	double stall=1;
	double frequency=atof(argv[2]); 
	double length;
	int int_len;
	double rand_max;
	double tmp=0;
	int i;

	if(argc == 4)
	{
	    period = atof(argv[3]);
	}

	if(frequency < 1)
	{
	    while((512./(frequency*stall)) > 512.)
	    {
		stall=stall+1;
	    } 

	    length=rnd(512./(frequency*stall));
	}
	else
	{

	    while((512.*(skip+1)/frequency) <= 512.)
	    {
		skip=skip+1;
	    }
    
	    length=rnd(512*skip/frequency);

	}

	int_len=length; 
	double x_values[int_len];
	double y_values[int_len];

	for(i=0;i<length;i++)
	{
	    tmp=tmp+period/length;    
	    x_values[i]=tmp;

	    if(!strncmp(argv[1],"sin",8))     
	    {
		y_values[i]=sin(x_values[i]);
		printf("%f\n",y_values[i]);
	    } 
    
	    if(!strncmp(argv[1],"cos",8))     
	    {
		y_values[i]=cos(x_values[i]);
		printf("%f\n",y_values[i]);
	    } 
    
	    if(!strncmp(argv[1],"rand",8))
	    {
		rand_max=RAND_MAX;
		srand(time(NULL)+i);
		y_values[i]=rand()/rand_max;
		printf("%f\n",y_values[i]);
	    }

	    if(!strncmp(argv[1],"zeros",8))     
	    {
		y_values[i]=0.0;
		printf("%f\n",y_values[i]);
	    } 

	    if(!strncmp(argv[1],"ones",8))     
	    {
		y_values[i]=1.0;
		printf("%f\n",y_values[i]);
	    } 
        
	    if(!strncmp(argv[1], "const", 8))
	    {
		y_values[i] = frequency;
		printf("%f\n", y_values[i]);
	    }
        
	    if(!strncmp(argv[1], "step",8))
	    {
		y_values[i] = i;
		printf("%f\n", y_values[i]);
	    }
        
	    if(!strncmp(argv[1], "pulse",8))
	    {
		if(i % (int) frequency == 0)
		    y_values[i] = period;
		else
		    y_values[i] = 0;
        
		skip = 1; stall = 1;
		printf("%f\n", y_values[i]);
	    }   
	}
    
	printf("%f\n",length);
	printf("%f\n",skip);
	printf("%f\n",stall);
	printf("%f\n",period);
    }


    return 1;
}


int rnd(double x)
{
    int y;

    if(x>0)
    {
        y=(int)(x+.5f);
    }
    else
    {
        y=(int)(x+.5f);
    }

    return y;
}


