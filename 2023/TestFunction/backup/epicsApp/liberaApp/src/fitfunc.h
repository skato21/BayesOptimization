#ifndef FITFUNC_H_
#define FITFUNC_H_

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
  {
#endif /* __cplusplus */

    extern void *new_object();
    extern void call_init(void *the_object, const char *smap);
    extern void call_fit( void *the_object, double *v1, double *v2, double *v3, double *v4, double *fitx, double *fity);
    extern void delete_object(void *the_object);

#ifdef __cplusplus

  }
#endif /* __cplusplus */
#endif /* FITFUNC_H_ */
