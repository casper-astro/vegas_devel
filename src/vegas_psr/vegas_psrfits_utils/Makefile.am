ACLOCAL_AMFLAGS = -I config

pkginclude_HEADERS = \
        fold.h \
        polyco.h \
        polyco_struct.h \
        psrfits.h 

noinst_HEADERS = 

dist_data_DATA = \
	guppi_PSRFITS_v3.4_fold_template.txt \
	guppi_PSRFITS_v3.4_search_template.txt

lib_LTLIBRARIES = libpsrfits_utils.la
libpsrfits_utils_la_SOURCES = \
         downsample.c \
         fold.c \
         misc_utils.c \
         polyco.c \
         read_psrfits.c \
         write_psrfits.c 

LDADD = libpsrfits_utils.la
AM_CPPFLAGS = -DPSRFITS_TEMPLATE_DIR='"$(datadir)"'

bin_PROGRAMS = fold_psrfits psrfits_singlepulse psrfits_subband

fold_psrfits_CFLAGS = @PTHREAD_CFLAGS@ $(AM_CFLAGS)
fold_psrfits_LDFLAGS = @PTHREAD_CFLAGS@ $(AM_LDFLAGS)
fold_psrfits_LDADD = libpsrfits_utils.la @PTHREAD_LIBS@

psrfits_subband_SOURCES = psrfits_subband.c \
			  psrfits_subband_cmd.h psrfits_subband_cmd.c
