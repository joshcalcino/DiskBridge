/* Date: Mar 4, 2015, author: Munan Gong
 * sundial.h
 * -------------------------------------------------------------------
 * Macros and functions that are useful in Sundial package.
 * Adopted from examples in cvode/cvRoberts_dns.*/

#ifndef SUNDIAL_H_
#define SUNDIAL_H_

#include <stdio.h>
#include <cvode/cvode.h> /* CV_SUCCESS */
#include <stdexcept> /*throw exceptions*/

/*check flags function for Sundial*/
void CheckFlag(const void *flagvalue, const char *funcname,
               const int opt);

#endif /*SUNDIAL_H_*/
