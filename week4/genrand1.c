/* COMP9334 Demonstration on 
   random number generation using C 
   
   Compile as: cc genrand1.c -o genrand1
   Usage: ./genrand1 

   C.T. Chou, 96/04/15
*/

#include <stdio.h>
#include <stdlib.h>

int main ()
{
  int i;

  for (i = 0; i < 10; i++)
      printf("%d\n",rand()); 
  
  return EXIT_SUCCESS; 

}
