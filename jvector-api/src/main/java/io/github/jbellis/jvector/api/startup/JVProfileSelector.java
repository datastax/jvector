package io.github.jbellis.jvector.api.startup;

import io.github.jbellis.jvector.api.types.JVMultiIndex;

import java.util.Optional;

public interface JVProfileSelector {

  /**
   <p>The user can specify a list of profiles to allow here.
   If the user wants to require a specific profile, then they simply
   <pre>{@code
    profileSelector.loadFirst(JVProfile.base_jvm);
   }</pre>
   </p>

   <p>If they want to prefer native mode, but allow the base mode should native mode fail to
   initialize, then they can instead do this:
   <pre>{@code
   profileSelector.loadFirst(JVProfile.native_ffi, JVProfile.base_jvm);
   }
   </pre></p>
   @param rankedProfiles the allowed profiles from {@link JVProfile}, in order of preference.
   @throws ProfileInitializationException if none of the profiles could be loaded.
   */
  Optional<JVMultiIndex> loadFirst(JVProfile... rankedProfiles);
}
