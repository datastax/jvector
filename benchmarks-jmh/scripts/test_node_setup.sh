###### Script for test node setup ######

sudo apt-get update

# Download JDK 22
wget https://download.java.net/java/GA/jdk22.0.2/c9ecb94cd31b495da20a27d4581645e8/9/GPL/openjdk-22.0.2_linux-x64_bin.tar.gz

# Extract JDK 22
tar -xzf openjdk-22.0.2_linux-x64_bin.tar.gz

sudo mkdir -p /usr/lib/jvm
sudo mv jdk-22.0.2 /usr/lib/jvm/jdk-22.0.2

########################################
# Setup Alternatives
########################################
sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk-22.0.2/bin/java" 1
sudo update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk-22.0.2/bin/javac" 1

########################################
# Verification
########################################

echo
echo "Installation complete. Current default Java version:"
java -version

# Install Maven
sudo apt-get install maven -y

# Install Git
sudo apt-get install git -y

# clone jvector
git clone https://github.com/datastax/jvector.git

# Build jvector
cd jvector
mvn clean install -DskipTests=true

# Run benchmarks
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar target/benchmarks-jmh-4.0.0-beta.3-SNAPSHOT.jar

