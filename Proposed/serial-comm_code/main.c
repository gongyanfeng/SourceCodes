#include <reg52.h>
#include <stdio.h>
#define uchar unsigned char
#define uint unsigned int

#define speed 100
//#define speed 50

uchar flag,a,i,circle;
uchar m2;
uint m1;
sbit beep=P2^3;
uchar CHAR_GET=0;

sbit I01 = P1^0;   //A1
sbit I02 = P1^1;   //B1
sbit I11 = P1^2;   //A2
sbit I12 = P1^3;   //B2
sbit PH1 = P1^4;	  //定义管脚
sbit PH2 = P1^5;
unsigned char code RUN[8]={0xf1,0xf3,0xf2,0xf6,0xf4,0xfc,0xf8,0xf9};  //步进电机相序表A-AB-B-BC-C-CD-D-DA
				     
unsigned char TableA[] = { 0XF7,0XFB,0XF3};	 //A线圈细分表
unsigned char TableB[] = { 0XeF,0XdF,0XcF};	 //B线圈细分表
void init()
{
	TMOD=0x20;
	TH1=0xfd;
	TL1=0xfd;
	TR1=1;
	REN=1;
	SM0=0;
	SM1=1;
	EA=1;
	ES=1;
	P0=0x00;
}

void delay(uint xms)
{
	for(m1=xms;m1>0;m1--)
      for(m2=110;m2>0;m2--);
}

/***************************************
函数功能：产生单相四拍脉冲控制步进机 2细分
**************************************/
void Go()
{	char i ,temp;

	for(circle=0;circle<6;circle++){
	     //A
	    PH1 = 0;  //PH1为0 则A线圈为反向电流
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableA[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }	
		PH2 = 0;  //PH2为0 则B线圈为反向电流
		I02 = 1;
		I12 = 1;   //输出0
		delay(speed);
		P0=0x80;
	
		//0
		PH1 = 0;  //PH1为0 则A线圈为反向电流
	    I01 = 1;  //输出0
		I11 = 1;
		PH2 = 1;  //PH2为1 则B线圈为正电流
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableB[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }
		delay(speed);
		P0=0xc0;
	
		//B
		PH1 = 1;   //PH1为1 则A线圈为正向电流
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableA[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }
		PH2 = 1;  //PH2为1 则B线圈为正向电流
		I02 = 1;  //输出0
		I12 = 1;	           ////
		delay(speed);
		P0=0xe0;
	
		//0
		PH1 = 1;   //PH1为1 则A线圈为正向电流
		I01 = 1;
		I11 = 1;
		PH2 = 0;   //PH2为0 则B线圈为反向电流
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableB[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }
		delay(speed);
		P0=0xf0;
	}
}

void  motor_ffw()
 { 
   unsigned char i;
  
      for (i=0; i<8; i++)       //一个周期八拍转动一个步距角14.4度
        {
          P1 = RUN[i]&0x1f;     //取数据
          //delay(100);             //调节转速
		  delay(20); //window下10ms,linux下20ms
        }
}

void main()
{
	char i ,temp;

	init();
	while(1){
		if(flag==1){
			ES=0;
			switch(CHAR_GET){
				case 1:  //A
					//puts("1-\n");
					beep=0;
					delay(2000);	
					beep=1;
					break;
				case 2:	//B
					P0=0x00;		
					 //linux上，delay(20ms)时，循环5次转一圈
					for(i=0;i<1;i++){//window调试助手上测试：delay(5ms)每八拍40ms时，25一圈;delay(10ms)时，5一圈
					   motor_ffw(); //调用旋转处理函数
					}
					P1=0x00;
					break;
				default:
					break;

			}
			SBUF=a;
			flag=0;

			while(!TI);
			TI=0;
			P0=0x01;		
			ES=1;	
			a=0;
		}	
	}
}
void ser() interrupt 4
{
	RI=0;
	a=SBUF;
	flag=1;
	//a = a & 0x0f;	
	if(a==0x41){//A
		CHAR_GET = 1;
	}
		
	else if(a==0x42){ //B
		CHAR_GET = 2;
	}
	else{
		CHAR_GET = 0;
	}
}