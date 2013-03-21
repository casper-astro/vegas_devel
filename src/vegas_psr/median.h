/* median.h: Header file for median.c */

typedef float elem_type ;

#define ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }
elem_type kth_smallest(elem_type a[], long int n, long int k);

#define median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

