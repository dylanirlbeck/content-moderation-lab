---
layout: about
title: project
permalink: /project
index: 2
nav: true
---

<h1>Project</h1>

**Ethical Moderation** is a programming-based ethics project for students, by students. The project
is completely open-source by design: we want to be part of a larger wave of revamped ethics education in computer science. We encourage instructors to incorporate this lab into their course; we challenge students to improve their ethical capacities by engaging with the project’s material; and, more broadly, we welcome anyone curious or concerned with content moderation to think hard about how it can and should be done.

<h4>Table of Contents</h4>
<div id="toc"></div>

## Background

**Content moderation** is the act of vetting user-generated content (UGC) that is regarded as irrelevant, obscene, illegal,
or otherwise as inappropriate for a given Internet service. In most cases, there is a central body that sets guidelines
for Although major social media sites like Facebook and Twitter have brought content moderation into the mainstream,
there are several other types of services that enforce guidelines on UGC: Internet forums, blogs, and news sites also
moderate content in increasingly unique ways. In short, content moderation happens in more places than you may expect.

At the same time, content moderation is fraught with controversy. With opaque and ever-changing guidelines, and relatively
little standardization across services, users can be easily confused about what types of posts are actually allowed on a given
service. A clear example of this is moderation of hate speech. In 2017, the New York Times [published a quiz](https://www.nytimeas.com/interactive/2017/10/13/technology/facebook-hate-speech-quiz.html)
that tested readers’ ability to “correctly” label content as meeting Facebook’s criteria for hate speech. The results are startling. In one case,
the given statement did not meet the criteria for hate speech, but in a follow-up question 92% of respondents said that they
did consider the statement to be hateful. This dissonance between what users think is allowed or not and what Facebook actually
permits is not only shocking, but dangerous: bad actors [can take advantage](https://www.propublica.org/article/facebook-enforcement-hate-speech-rules-mistakes) of inconsistencies to post harmful content.

To many, the content moderation problem seems easy: just have humans evaluate each piece of potentially harmful content objectively.
However, this approach runs into two main problems. First, the massive scale of companies like Facebook or Twitter means that there
are millions of controversial posts to comb through. The amount of human moderators required to perform fully-manual checking would
be too large for any one organization. At the same time, there are posts that [slip through the cracks](https://www.wired.com/story/qanon-conspiracy-facebook-meme-ai/) in a fully automated system. There
is a tension between human and automated content moderation, and it doesn’t seem like any platform has found the right balance.

But given its impact on society, the mechanics of content moderation must be constantly discussed. And given how many future computer
scientists will work on services that allow for UGC, it is imperative that ethics courses cover content moderation.

## Learning Objectives

**Ethical Moderation** is designed to have students grapple with the ethical issues at play in content moderation. It is a programming-based
ethics assignment, designed to be completed individually. That said, we provide various questions -- after each programming section and upon completion
of the lab -- to be discussed in groups or through short papers. We _highly_ recommend that instructors include these verbal and/or written components
in the assigned work for this project. If you're completing this lab individually, we recommend taking time to reflect on the questions and answer them for yourself; everyone approaches ethical dilemmas with a unique
perspective, but it is important to be able to articulate where you stand and why.

Keeping this in mind, the **learning objectives** for this project are:

- Learn different ways in which content moderation can be performed and
- Compare and contrast the efficacy of each approach, from a technical, and ethical perspective.
- Articulate when and why human moderation is needed, and understand the
  tradeoffs between automated and human review.
- Consider the challenges that modern platforms face when performing content moderation.
- Understand the impact that content moderation (or lack thereof) has on users.

## Task

_Sport-It_ is a popular social media platform known for its unique, sports-based community. Over the last few months,
_Sport-It_ has been rapidly growing in scale: the daily active users (DAUs) increased by 200% in the last month alone!
As a result, there has been an explosion in the number of posts made on the platform. It is untenable for the moderators
at _Sport-It_ to continue conducting manual review of posts, and so they’ve hired you to design an automated content moderation system.

These moderators are very intentional about their moderation: they only allow posts about sports. Topics like politics, finance,
and cooking, for example, are not allowed on _Sport-It_. They’ve directed you to write software that removes all posts not directly
related to sports.

### Part 0: Grab the Code

The code for this lab can be found in [here](https://github.com/dylanirlbeck/hackillinois-2021). The relevant code is contained inside a Jupyter
Notebook, and we provide several utilities to do probability calculation and
accuracy checking.

### Part 1: "Naive" Moderation

Your first task is to use Naive Bayes, a simple and well-known machine learning algorithm, to classify users’ posts as either
on-topic or off-topic. The reviewers at _Sport-It_ figured you’d use some sort of machine learning, so they’ve provided you with
a corpus of posts and the corresponding labels for the posts. They also wanted to verify that your algorithm was working properly
before deploying it on their site, so they’ve created a separate corpus of unlabeled posts to test your algorithm on.

In short, your algorithm will, given a users' post, determine whether the probability
that the post is on-topic or the probability that it is off-topic is higher. In math,

$$P(Type = On|Post) > P(Type = Off|Post)$$

or

$$P(Type = On|Post) \le P(Type = Off|Post)$$

Where $$On$$ represents an on-topic post and $$Off$$ an off-topic post (the
$$|$$ should be read as "given").

In order to calculate and subsequently compare these values, you need to multiply the probabilities that each individual word of the
post shows up in a valid and invalid post. We determine these probabilities as follows:

$$P(Type = On|Post) = P(Type = On) \prod_{All\ words} P(Word | Type = On)$$

and

$$P(Type = Off|Post) = P(Type = Off) \prod_{All\ words} P(Word | Type = Off)$$

Here,
$$P(Type = On)$$
and
$$P(Type = Off)$$
are just the probabilities that a given post will be on or off, respectively.
These probabilities are provided for you in the starter code.

The post will then be labeled as on- or off-topic depending on which of the two
probabilities are higher.

**Once you've finished this part, your content moderation accuracy should be at about 90*%.***

### Part 2: The Consequences of Naivety

The algorithm you have just implemented is known as a **bag-of-words** algorithm. Notice that our algorithm only accounts for the
presence of words. It does not account for any order or structure the post may have. Simply put, we consider each post as a
random collection, or bag, of words, rather than a structure of words. There are some major downsides to just considering the
amount of times a word appears in a certain label, which we will explore soon.

You should have begun to realize that all invalid posts generally have a common theme in their text. Most of them may contain words
such as “politics”, “food”, “religion”, and more. However, there may be some cases where we misinterpret some posts as invalid, when
in actuality, many would agree that it should be valid. Let’s take a look at one example.

`Colin Kaepernick Signs a 6 Year Contract`

This post is clearly about sports; it is simply stating that Colin Kaepernick, an NFL quarterback, signed a contract. However, our naive
moderation algorithm will likely choose to remove this post. Why does this happen? The answer lies in the training data. Remember, _SportIt_ provided
you with a training corpus of off-topic posts. Many of these posts discussed politics, and Colin Kaepernick has been a controversial figure due to his history of
kneeling for the national anthem.

**Question #1: How may content moderation algorithms may create involuntary censorship, and who it might effect in the case of _SportIt_?**

### Part 3: Introducing Human Content Moderation

It is now time to modify the naive content moderation algorithm to better handle “edge cases”; that is, posts which have aspects of
being on-topic and off-topic. SportIt moderators want you to specifically gather posts whose difference in probability falls within
a given threshold and send these posts back to the moderators for human review. However, they want you to decide what this threshold
value should be.

For clarity, the threshold is defined as:
$$|P(Type = On|Post) - P(Type = Off|Post)|$$

Play around with your chosen threshold value and analyze what posts are
considered as "edge cases".

**Question #1: When it is appropriate to introduce human
reviewers into content moderation?**

**Question #2: Would you consider any of the "edge cases" that your algorithm
returned to not be edge cases? Why?**

## Discussion Questions

Some example questions that could follow the programming component are:

- In many cases, content could fall into the valid and invalid category. For example, a post about Colin Kapernick could also talk about kneeling for the national anthem. How should cases where there is no “clear” answer be handled?
- How do you balance human review in content moderation? How, if at all, does human review differ if you are a big company or a small one?
- Is there an “optimal” way to moderate content? Are there any best practices that any company maintaining UGC could apply?
- To what degree should speech be moderated? Should there be legal protections for speech on these platforms?

## FAQ

- **Why did you not use real Facebook/Twitter data?**

  - As we were constructing this assignment, we had to consider what was most appropriate for our audience. At a
    university, and especially at a large one like UIUC, students come from all walks of life. A significant portion
    of the content on Facebook -- and particularly the types of posts we’d need to include in our dataset -- would likely
    be extremely graphic, offensive, or both. We concluded that in order to provide all students with an optimal and safe
    learning experience, we should aim for our dataset and assignment to be neutral, while still being as illustrative as
    possible. We hope that we achieve this balance, but if you feel there is a different way we could set up the assignment,
    please let us know!

- **How did you gather the project data?**
  - The data is pulled from Reddit, specifically from the subreddits r/sports and r/politics. To
    create the dataset, we merged a subset of posts from each subreddit
    together, assigning on-topic labels for the posts from r/sports and
    off-topic labels to the posts from r/politics. As mentioned earlier, there
    are a number of interesting posts that may be labeled as off-topic, but may
    actually be on-topic for _SportIt_(and vice-versa for posts labeled as
    on-topic) -- it is our hope that students see
    these interesting patterns and think critically about the effects of
    either removing posts from a platform or not.
