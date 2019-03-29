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
sbit PH1 = P1^4;	  //����ܽ�
sbit PH2 = P1^5;
unsigned char code RUN[8]={0xf1,0xf3,0xf2,0xf6,0xf4,0xfc,0xf8,0xf9};  //������������A-AB-B-BC-C-CD-D-DA
				     
unsigned char TableA[] = { 0XF7,0XFB,0XF3};	 //A��Ȧϸ�ֱ�
unsigned char TableB[] = { 0XeF,0XdF,0XcF};	 //B��Ȧϸ�ֱ�
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
�������ܣ�������������������Ʋ����� 2ϸ��
**************************************/
void Go()
{	char i ,temp;

	for(circle=0;circle<6;circle++){
	     //A
	    PH1 = 0;  //PH1Ϊ0 ��A��ȦΪ�������
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableA[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }	
		PH2 = 0;  //PH2Ϊ0 ��B��ȦΪ�������
		I02 = 1;
		I12 = 1;   //���0
		delay(speed);
		P0=0x80;
	
		//0
		PH1 = 0;  //PH1Ϊ0 ��A��ȦΪ�������
	    I01 = 1;  //���0
		I11 = 1;
		PH2 = 1;  //PH2Ϊ1 ��B��ȦΪ������
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableB[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }
		delay(speed);
		P0=0xc0;
	
		//B
		PH1 = 1;   //PH1Ϊ1 ��A��ȦΪ�������
		for(i = 0; i<3; i++)
		{  temp = P1;
		   P1 = TableA[i]; 
		   P1 = P1&temp; 
		   delay(1);
		 }
		PH2 = 1;  //PH2Ϊ1 ��B��ȦΪ�������
		I02 = 1;  //���0
		I12 = 1;	           ////
		delay(speed);
		P0=0xe0;
	
		//0
		PH1 = 1;   //PH1Ϊ1 ��A��ȦΪ�������
		I01 = 1;
		I11 = 1;
		PH2 = 0;   //PH2Ϊ0 ��B��ȦΪ�������
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
  
      for (i=0; i<8; i++)       //һ�����ڰ���ת��һ�������14.4��
        {
          P1 = RUN[i]&0x1f;     //ȡ����
          //delay(100);             //����ת��
		  delay(20); //window��10ms,linux��20ms
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
					 //linux�ϣ�delay(20ms)ʱ��ѭ��5��תһȦ
					for(i=0;i<1;i++){//window���������ϲ��ԣ�delay(5ms)ÿ����40msʱ��25һȦ;delay(10ms)ʱ��5һȦ
					   motor_ffw(); //������ת������
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