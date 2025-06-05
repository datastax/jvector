package io.github.jbellis.jvector.api.types;

import java.util.List;
import java.util.Vector;

/**
 An indexer is how you mutably extend the contents of an index.
 */
public interface JVIndexer {
  /**
   <P>Add vectors to a mutable indexer at the given offset.
   While it is not useful to add zero vectors, this case is not specifically wrong in terms of
   how other APIs may call this method, and no error is thrown in such case.</P>
   <HR/>
   <P>Examples</P>
   <PRE>{@code

     // add a single vector, starting at ordinal 0
     indexer.add(0,Vector.of(1.0,2.0,3.0));

     // add a couple vectors, starting at ordinal 10
     indexer.add(10,
       Vector.of(1.0,2.0,3.0),
       Vector(-1.0,-2.0,-3.0)
     );

     // bulk add vectors, starting at ordinal 100
     Vector<?>[] batch = …;
     indexer.add(100,batch);
   }
   </PRE>
   * @param baseOrdinal The first (inclusive) index of the first vector in the varags list.
   * @param vectors A varargs list of vectors to add to the index. This can be a single vector,
   * multiple vector arguments, or an array)
   */
  void add(int baseOrdinal, Vector<?>... vectors);
  void remove(int... ordinals);
  JVIndexer consolidate();
}
