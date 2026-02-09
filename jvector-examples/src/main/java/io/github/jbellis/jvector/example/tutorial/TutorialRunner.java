package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;

public class TutorialRunner {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please pick an example");
        }
        String[] forwardArgs = new String[args.length - 1];
        for (int i = 0; i < forwardArgs.length; i++) {
            forwardArgs[i] = args[i + 1];
        }

        switch (args[0]) {
            case "intro":
                VectorIntro.main(forwardArgs);
                break;
            case "disk":
                DiskIntro.main(forwardArgs);
                break;
            case "ltm":
                LargerThanMemory.main(forwardArgs);
                break;
            default:
                throw new IllegalArgumentException("Unknown example" + args[0]);
        }
    }
}
