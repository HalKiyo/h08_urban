###############################################################################################################
# h08 environment
###############################################################################################################
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

RUN apt clean && \
    apt update && \
    yes | apt upgrade && \
    apt install -y \
    g++ \
    gcc \
    gfortran \
    cmake \
    curl \
    build-essential \
    libcurl4-openssl-dev \
    libnetcdf-dev \
    gdal-bin \
    libgdal-dev \
    libfftw3-dev \
    libpcre3-dev \
    liblapack-dev \
    libblas-dev \
    libglib2.0-dev \
    ghostscript \
    ghostscript-x \
    imagemagick \
    ffmpeg \
    xdg-utils \
    wget \
    eog \
    git \
    sudo \
    vim \
    bc \
    powerline \
    bzip2 \
    unzip \
    file \
    xz-utils \
    python3.10 \
    python3.10-dev \
    python3-tk \
    python3-pip \
    python3-opencv \
    libgeos-dev

# remove cache files
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*

# build netcdf with gcc and g-fortran
ENV FC=gfortran
ENV CC=gcc

# set library location
ENV PREFIXDIR=/usr/local

# zlib install
WORKDIR /
ARG ZLIB_VERSION=zlib-1.2.11
RUN wget https://www.zlib.net/fossils/${ZLIB_VERSION}.tar.gz \
    && tar -xvzf ${ZLIB_VERSION}.tar.gz
RUN cd ${ZLIB_VERSION} \
    && ./configure --prefix=${PREFIXDIR} \
    && make \
    && make check \
    && make install
WORKDIR /
RUN rm -rf ${ZLIB_VERSION}.tar.gz ${ZLIB_VERSION}

# szip install
WORKDIR /
ARG SZIP=2.1.1
ARG SZIP_VERSION=szip-2.1.1
RUN wget https://support.hdfgroup.org/ftp/lib-external/szip/${SZIP}/src/${SZIP_VERSION}.tar.gz \
    && tar -xvzf ${SZIP_VERSION}.tar.gz
RUN cd ${SZIP_VERSION} \
    && ./configure --prefix=${PREFIXDIR} \
    && make \
    && make check \
    && make install
WORKDIR /
RUN rm -rf ${SZIP_VERSION}.tar.gz ${SZIP_VERSION}

# HDF5 install
WORKDIR /
ARG HDF5=hdf5-1.14
ARG HDF5_VERSION=hdf5-1.14.0
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/${HDF5}/${HDF5_VERSION}/src/${HDF5_VERSION}.tar.gz \
    && tar -xvzf ${HDF5_VERSION}.tar.gz
RUN cd ${HDF5_VERSION} \
    && ./configure --prefix=${PREFIXDIR} --enable-fortran --with-szlib=${PREFIXDIR} --with-zlib \
    && make \
    && make check \
    && make install
WORKDIR /
RUN rm -rf ${HDF5_VERSION}.tar.gz ${HDF5_VERSION}
ENV LD_LIBRARY_PATH="${PREFIXDIR}/&{HDF5_VERSION}/lib:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${PREFIXDIR}/${HDF5_VERSION}/lib:${LIBRARY_PATH}"
ENV PATH="${PREFIXDIR}/${HDF5_VERSION}/lib:${PATH}"

# netcdf-c install
WORKDIR /
ARG NETCDFC=4.9.2
ARG NETCDFC_VERSION=netcdf-c-4.9.2
RUN wget https://downloads.unidata.ucar.edu/netcdf-c/${NETCDFC}/${NETCDFC_VERSION}.tar.gz \
    && tar -xvzf ${NETCDFC_VERSION}.tar.gz
ENV LD_LIBRARY_PATH="${PREFIXDIR}/&{HDF5_VERSION}/lib:${LD_LIBRARY_PATH}"
ENV CPPFLAGS="-I/${PREFIXDIR}/include -I/${PREFIXDIR}/${HDF5_VERSION}/include"
ENV LDFLAGS="-L/${PREFIXDIR}/lib -L/${PREFIXDIR}/${HDF5_VERSION}/lib"
RUN cd ${NETCDFC_VERSION} \
    && ./configure --prefix=${PREFIXDIR} --enable-netcdf-4 --enable-shared --enable-dap --disable-dap-remote-tests \
    && make \
#&& make check \
    && make install
WORKDIR /
RUN rm -rf ${NETCDFC_VERSION}.tar.gz ${NETCDFC_VERSION}

# netcdf-fortran install
WORKDIR /
ARG NETCDF=4.6.0
ARG NETCDFF_VERSION=netcdf-fortran-4.6.0
RUN wget https://downloads.unidata.ucar.edu/netcdf-fortran/${NETCDF}/${NETCDFF_VERSION}.tar.gz \
    && tar -xvzf ${NETCDFF_VERSION}.tar.gz
RUN cd ${NETCDFF_VERSION} \
    && ./configure --prefix=${PREFIXDIR} \
    && make \
#&& make check \
    && make install
WORKDIR "/usr/local/bin"
WORKDIR /
RUN rm -rf ${NETCDFF_VERSION}.tar.gz ${NETCDFF_VERSION}
ENV NETCDF_ROOT="${PREFIXDIR}/${NETCDFC_VERSION}"
ENV NETCDF_F_ROOT="${PREFIXDIR}/${NETCDFF_VERSION}"
ENV NETCDF="${NETCDF_ROOT}"

# GMT install
WORKDIR /
ARG GMT_VERSION=6.4.0
ARG GSHHG_VERSION=2.3.7
ARG DCW_VERSION=2.1.1
RUN wget "https://github.com/GenericMappingTools/gmt/releases/download/${GMT_VERSION}/gmt-${GMT_VERSION}-src.tar.gz" -P /tmp/
RUN wget "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/${GSHHG_VERSION}/gshhg-gmt-${GSHHG_VERSION}.tar.gz" -P /tmp/
RUN wget "https://github.com/GenericMappingTools/dcw-gmt/releases/download/${DCW_VERSION}/dcw-gmt-${DCW_VERSION}.tar.gz" -P /tmp/
COPY ./build_gmt.sh /tmp
RUN bash /tmp/build_gmt.sh

# copy .bashrc
COPY ./alias /root/.bashrc

# copy vimfiles
COPY ./vimrc-script /root/.vimrc
RUN mkdir /root/.vim

###############################################################################################################
# python environment
###############################################################################################################
# update pip
RUN python3 -m pip install --upgrade pip

# install python modules
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# pythorch is installed manually. Not with requirements.txt
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121/

# upgrading numpy from 1.17.4 to 1.23.5
RUN pip install --upgrade numpy

# downgrading shapely from 1.8.5.post to 1.7.1
RUN yes | pip uninstall shapely
RUN pip install shapely==1.7.1

# gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN yes | sudo apt update \
    && yes | sudo apt upgrade \
    && yes | sudo apt install libpq5 \
    && yes | sudo apt install libpq-dev \
    && yes | sudo apt install gdal-bin \
    && yes | sudo apt install libgdal-dev

###############################################################################################################
# not neccesary
###############################################################################################################

# install nvm/Node.js
RUN mkdir /usr/local/nvm
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 16.13.0
RUN curl https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# install plug.vim
WORKDIR "/root/.vim/autoload"
RUN curl -fLO https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# install coc.nvim extensions
RUN mkdir -p "/root/.config/coc/extensions"
WORKDIR "/root/.config/coc/extensions"
RUN if [ ! -f package.json ] ; then echo '{"dependencies": {}}' > package.json ; fi && \
    npm install \
    coc-json \
    --global-style --ignore-scripts --no-bin-links --no-package-lock --only=prod

