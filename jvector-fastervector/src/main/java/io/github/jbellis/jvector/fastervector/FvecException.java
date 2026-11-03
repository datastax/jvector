package io.github.jbellis.jvector.fastervector;

/**
 * Unchecked exception thrown when a native fvec C-API function returns a non-zero
 * result code.  The native side prints details to stderr; the message in this
 * exception names the C function that failed.
 */
public class FvecException extends RuntimeException {

    public FvecException(String message) {
        super(message);
    }

    public FvecException(String message, Throwable cause) {
        super(message, cause);
    }
}
