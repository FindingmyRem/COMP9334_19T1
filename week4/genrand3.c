/* COMP9334 Demonstration on 
   random number generation using C 
   
   Compile as: cc genrand3.c -o genrand3
   Usage: ./genrand3  

   C.T. Chou, 06/04/15
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

int main ()
{
  int i;

   // Use the current time as the seed 
   srand(time(NULL));

   // Generate the random numbers 
   for (i = 0; i < 10; i++)
      printf("%d\n",rand()); 
  
   return EXIT_SUCCESS; 

}
