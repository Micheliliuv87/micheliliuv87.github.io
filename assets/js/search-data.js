// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Current work",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-distance-measures-for-data-science",
        
          title: "Distance Measures for Data Science",
        
        description: "Include many distance measures: which come in handy and help me through many of my data science projects",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/distance-measure/";
          
        },
      },{id: "post-winning-with-data-goizueta-msba-students-shine-in-travelers-university-modeling-competition-voice-of-goizueta",
        
          title: 'Winning with Data: Goizueta MSBA Students Shine in Travelers University Modeling Competition -... <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "Goizueta MSBA students excelled in the Travelers University Modeling Competition, advancing to the semifinals with their innovative predictive model. Read about their experience and key takeaways.",
        section: "Posts",
        handler: () => {
          
            window.open("https://www.voiceofgoizueta.com/winning-with-data-goizueta-msba-students-shine-in-travelers-university-modeling-competition/", "_blank");
          
        },
      },{id: "post-autoencoder-basics-and-how-to-implement",
        
          title: "Autoencoder Basics and How to Implement",
        
        description: "Simple implementation of autoencoder framework",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/distance-measure-copy/";
          
        },
      },{id: "post-practical-applications-of-the-bag-of-words-model",
        
          title: "Practical Applications of the Bag of Words Model",
        
        description: "Application of Bag of words, examples of TF-IDF, Text Classification, and Sentiment Analysis",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/bag-of-words-2/";
          
        },
      },{id: "post-the-bag-of-words-model-a-comprehensive-analysis-of-nlp-39-s-foundational-technique",
        
          title: "The Bag of Words Model, A Comprehensive Analysis of NLP&#39;s Foundational Technique",
        
        description: "BoW introduction, history, and use cases",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/bag-of-words-1/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-graduated-from-emory-university",
          title: 'Graduated From Emory University',
          description: "",
          section: "News",},{id: "news-upcomming-updates",
          title: 'Upcomming Updates',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-check-out-my-post-distance-measure-for-data-science-a-useful-note-to-understand-distance-measures-and-when-to-use-them",
          title: 'Check out my post — Distance Measure for Data Science, a useful note...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%69%63%68%65%6C%69.%6C%69%75@%61%6C%75%6D%6E%69.%65%6D%6F%72%79.%65%64%75", "_blank");
        },
      },{
        id: 'social-discord',
        title: 'Discord',
        section: 'Socials',
        handler: () => {
          window.open("https://discord.com/users/mickyv87", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Micheliliuv87", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/micheli87", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/Mickyv87", "_blank");
        },
      },{
        id: 'social-wechat_qr',
        title: 'Wechat_qr',
        section: 'Socials',
        handler: () => {
          window.open("", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
