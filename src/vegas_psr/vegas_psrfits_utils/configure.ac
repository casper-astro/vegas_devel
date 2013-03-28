AC_INIT([psrfits_utils], [1.0])
AC_CONFIG_SRCDIR([downsample.c])
AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_AUX_DIR([config])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_PROG_CC
AC_PROG_LIBTOOL

AX_PTHREAD

AC_CHECK_LIB([m],[sin])
AC_CHECK_LIB([cfitsio],[ffgky],[],
	     [AC_MSG_ERROR([cfitsio is required])])

AC_ARG_ENABLE([sse],
	      [AS_HELP_STRING([--enable-sse], 
			      [enable SSE folding code])],
	      [],
	      [enable_sse=no])
if test "x$enable_sse" = "xyes"; then
	AC_DEFINE([FOLD_USE_INTRINSICS], [1], [Use SSE intrinsics for folding])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
