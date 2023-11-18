#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <iostream>
#include <vector>

class Application {
public:
	auto run() -> void {
		this->init();

		this->mainLoop();
	}

private:
	SDL_Window* window = nullptr;
	VkInstance instance = VK_NULL_HANDLE;

	//std::string errorMessage;
	std::string appName = "Vulkanus";
	std::string engineName = "V-TWIN";

	bool quit = false;

	/* 
			initialization and create vulkan instance
	*/
	auto init() -> void {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error(SDL_GetError());

		this->window = SDL_CreateWindow(
			appName.c_str(),				// title
			SDL_WINDOWPOS_CENTERED, // x position
			SDL_WINDOWPOS_CENTERED, // y position
			600, 600,				// width, height value
			SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN	// window flag
		);

		createInstance();
	}

	auto createInstance() -> void {
		auto extensionCount = uint32_t(0);
		SDL_Vulkan_GetInstanceExtensions(this->window, &extensionCount, nullptr);

		auto extensionNames = std::vector<const char*>(extensionCount);
		SDL_Vulkan_GetInstanceExtensions(this->window, &extensionCount, extensionNames.data());

		auto appInfo = VkApplicationInfo {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = appName.c_str(),
			.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
			.pEngineName = engineName.c_str(),
			.engineVersion = VK_MAKE_VERSION(0, 0, 1),
			.apiVersion = VK_API_VERSION_1_3
		};

		auto createInfo = VkInstanceCreateInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = 0,		// validation layer size
			.ppEnabledLayerNames = nullptr,
			.enabledExtensionCount = static_cast<uint32_t>(extensionNames.size()),
			.ppEnabledExtensionNames = extensionNames.data()
		};

		vkCreateInstance(&createInfo, nullptr, &(this->instance));
	}

	auto mainLoop() -> void {
		auto event = SDL_Event();

		while (quit == false) {
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					this->quit = true;
					break;
				}
			}
		}
	}

};

auto main(int argc, char* argv[]) -> int {
	auto app = Application();

	try {
		app.run();
	}
	catch (std::exception exp) {
		std::cout << exp.what() << std::endl;
	}

	return 0;
}