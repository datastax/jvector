package io.github.jbellis.jvector.api.startup;

/**
 This error is thrown when there is a problem initializing a profile which is required by the user.
 Ideally, this error collects a list of profile initialization exceptions which occured in the
 case that the user specified multiple profiles and none of them succeeded.
 */
public class ProfileInitializationException extends RuntimeException {
  public ProfileInitializationException(String message) {
    super(message);
  }

  public ProfileInitializationException(String message, Throwable cause) {
    super(message, cause);
  }

}
