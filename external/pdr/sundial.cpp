#include "sundial.h"

#include <string>

void CheckFlag(const void *flagvalue, const char *funcname, 
               const int opt) {
   int *errflag;

   /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
   if (opt == 0 && flagvalue == NULL) {
     fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
         funcname);
     throw std::runtime_error("SUNDIALS:Sundials error.");
     return; 
   }

   /* Check if flag < 0 */
   else if (opt == 1) {
     errflag = (int *) flagvalue;
     if (*errflag < 0) {
       fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
           funcname, *errflag);
       throw std::runtime_error("SUNDIALS:Sundials error.");
       return; 
     }
   }

   /* Check if function returned NULL pointer - no memory allocated */
   else if (opt == 2 && flagvalue == NULL) {
     fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
         funcname);
     throw std::runtime_error("SUNDIALS:Memory error.");
     return; 
   }

   /* Check if CV_SUCCESS for integration. */
   else if (opt == 3) {
     errflag = (int *) flagvalue;
     if (*errflag != CV_SUCCESS) {
       throw std::runtime_error(
           std::string("SUNDIALS:") + funcname +
           " failed with flag = " + std::to_string(*errflag));
       return; 
     }
   }
}
