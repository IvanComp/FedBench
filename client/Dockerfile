# Build stage for qemu-system-arm
FROM debian:stable-slim AS qemu-builder
ARG QEMU_VERSION=6.0.0
ENV QEMU_TARBALL="qemu-${QEMU_VERSION}.tar.xz"
WORKDIR /qemu

# Update package lists
RUN apt-get update

# Pull source
RUN apt-get -y install wget xz-utils
RUN wget "https://download.qemu.org/${QEMU_TARBALL}"

# Verify signatures
RUN apt-get -y install gnupg
RUN wget "https://download.qemu.org/${QEMU_TARBALL}.sig"
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys CEACC9E15534EBABB82D3FA03353C9CEF108B584
RUN gpg --verify "${QEMU_TARBALL}.sig" "${QEMU_TARBALL}"

# Extract source tarball
RUN tar xvf "${QEMU_TARBALL}"

# Build source
RUN apt-get -y install python3 build-essential libglib2.0-dev libpixman-1-dev ninja-build
RUN apt-get -y install libfdt-dev zlib1g-dev
RUN apt-get -y install flex bison
RUN ./qemu-${QEMU_VERSION}/configure --static --target-list=arm-softmmu,aarch64-softmmu
RUN make -j$(nproc)

# Strip the binary
RUN strip "arm-softmmu/qemu-system-arm" "aarch64-softmmu/qemu-system-aarch64" "qemu-img"

# Build stage for fatcat
FROM debian:stable-slim AS fatcat-builder
ARG FATCAT_VERSION=v1.1.0
ARG FATCAT_CHECKSUM="303efe2aa73cbfe6fbc5d8af346d0f2c70b3f996fc891e8859213a58b95ad88c"
ENV FATCAT_TARBALL="${FATCAT_VERSION}.tar.gz"
WORKDIR /fatcat

# Update package lists
RUN apt-get update

# Pull source
RUN apt-get -y install wget
RUN wget "https://github.com/Gregwar/fatcat/archive/${FATCAT_TARBALL}"
RUN echo "${FATCAT_CHECKSUM} ${FATCAT_TARBALL}" | sha256sum --check

# Extract source tarball
RUN tar xvf "${FATCAT_TARBALL}"

# Build source
RUN apt-get -y install build-essential cmake
RUN cmake fatcat-* -DCMAKE_CXX_FLAGS='-static'
RUN make -j$(nproc)

# Build the dockerpi VM image
FROM debian:stable-slim AS dockerpi-vm
LABEL maintainer="Luke Childs <lukechilds123@gmail.com>"
ARG RPI_KERNEL_URL="https://github.com/dhruvvyas90/qemu-rpi-kernel/archive/afe411f2c9b04730bcc6b2168cdc9adca224227c.zip"
ARG RPI_KERNEL_CHECKSUM="295a22f1cd49ab51b9e7192103ee7c917624b063cc5ca2e11434164638aad5f4"

COPY --from=qemu-builder /qemu/arm-softmmu/qemu-system-arm /usr/local/bin/qemu-system-arm
COPY --from=qemu-builder /qemu/aarch64-softmmu/qemu-system-aarch64 /usr/local/bin/qemu-system-aarch64
COPY --from=qemu-builder /qemu/qemu-img /usr/local/bin/qemu-img
COPY --from=fatcat-builder /fatcat/fatcat /usr/local/bin/fatcat

RUN apt-get update && \
    apt-get install -y wget unzip && \
    rm -rf /var/lib/apt/lists/*

ADD $RPI_KERNEL_URL /tmp/qemu-rpi-kernel.zip

RUN cd /tmp && \
    echo "$RPI_KERNEL_CHECKSUM  qemu-rpi-kernel.zip" | sha256sum -c && \
    unzip qemu-rpi-kernel.zip && \
    mkdir -p /root/qemu-rpi-kernel && \
    cp qemu-rpi-kernel-*/kernel-qemu-4.19.50-buster /root/qemu-rpi-kernel/ && \
    cp qemu-rpi-kernel-*/versatile-pb.dtb /root/qemu-rpi-kernel/ && \
    rm -rf /tmp/*

VOLUME /sdcard

ADD ./entrypoint.sh /entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

# Build the dockerpi image
FROM dockerpi-vm as dockerpi
LABEL maintainer="Luke Childs <lukechilds123@gmail.com>"
ARG FILESYSTEM_IMAGE_URL="http://downloads.raspberrypi.org/raspbian_lite/images/raspbian_lite-2019-09-30/2019-09-26-raspbian-buster-lite.zip"
ARG FILESYSTEM_IMAGE_CHECKSUM="a50237c2f718bd8d806b96df5b9d2174ce8b789eda1f03434ed2213bbca6c6ff"

ADD $FILESYSTEM_IMAGE_URL /filesystem.zip

# Copy the requirements file
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

RUN echo "$FILESYSTEM_IMAGE_CHECKSUM  /filesystem.zip" | sha256sum -c